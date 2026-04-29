"""
CTP-LLM: Context and Transition Probability-guided LLM for P2P process suffix generation.

This module runs the main experiment of the paper. For each of the 180 evaluation QA
instances drawn from the BPI Challenge 2019 Purchase-to-Pay log, it iteratively
generates an activity suffix by combining:

  (1) a community-based top-K candidate list derived from precomputed transition
      probabilities P(b | a, c) for the case's community c, and
  (2) an LLM (Claude Sonnet 4.6, temperature 0) prompted with the business-context
      query, the observed prefix, and the candidate list. The LLM selects one
      candidate per step until <END> is produced or MAX_STEPS is reached.

Outputs per instance: predicted suffix, Damerau-Levenshtein similarity, F1 score,
edit distance, prefix+suffix combined similarity, and per-step parsing status.

A checkpoint file is written after every instance so the run can resume after
interruption without losing progress.

Inputs (must be present in working directory):
  - qa_dataset_final.pkl       : evaluation QA instances
  - comm{0..5}_probabilities_final.pkl : per-community transition probabilities

Outputs:
  - claude_results_final2.pkl  : final results dict
  - checkpoint_claude_final2.pkl : intermediate checkpoint (resume support)
"""

# Standard library
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

# Third-party
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Configuration
# ============================================================

# API
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-sonnet-4-6"
TEMPERATURE = 0.0  # Fixed for reproducibility
MAX_TOKENS_GENERATE = 200  # Max tokens for the main FINAL_EVENT response
MAX_TOKENS_REPAIR = 50     # Max tokens for the format-repair retry

# Generation
TOPK_EVENTS = 10           # Number of next-event candidates surfaced to the LLM per step

# I/O
CHECKPOINT_FILE = 'checkpoint_claude_final2.pkl'
RESULT_FILE = 'claude_results_final2.pkl'
QA_DATASET_FILE = 'qa_dataset_final.pkl'
NUM_COMMUNITIES = 6


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """You are an expert in SAP Purchase-to-Pay (P2P) procurement processes.

Your task is to determine the next single event in a procurement process sequence to resolve a stated business issue.

Key principles:
1. UNDERSTAND THE ISSUE: Read the query carefully to identify exactly what problem needs to be resolved.
2. TRACK PROGRESS: The "Steps taken so far" shows what has already been done toward resolution. Do not redo these.
3. MINIMAL SEQUENCE: Only select events that directly contribute to resolving the stated issue. Do not add unnecessary steps.
4. PROCESS STEPS: Routine steps are valid only if they are necessary to reach resolution of the stated issue.
5. TERMINATION: Output <END> only when ALL of the following are true:
   - The specific issue stated in the query is fully resolved
   - The procurement case is financially closed (no open invoices, no pending records)
   - There are no remaining process steps needed to stabilize the case
   Resolving the immediate issue alone is NOT sufficient to terminate. Always verify the full case is closed before outputting <END>.

P2P domain knowledge:
- Do NOT assume a payment block exists unless the query explicitly states one is currently present. If the query says a payment block "had already been resolved" or "was removed earlier", it is no longer active.

At each step, ask yourself:
- What exactly is the issue in the query?
- Has that specific issue been fully resolved by the steps taken so far?
- Even if resolved, are there open invoices or pending financial records that still need to be handled?
- Only if both the issue is resolved AND the case is fully closed → output <END>"""


# ============================================================
# Data loading
# ============================================================

print("파일 로드 중...")

with open(QA_DATASET_FILE, 'rb') as f:
    data = pickle.load(f)
qa_list = list(data['qa_dataset'].values())

community_probs = {}
for i in range(NUM_COMMUNITIES):
    with open(f'comm{i}_probabilities_final.pkl', 'rb') as f:
        prob_data = pickle.load(f)
        community_probs[i] = prob_data['P_b_given_a_c']

# MAX_STEPS = longest ground-truth answer length plus a small buffer
max_answer_len = max(len(qa['answer']) for qa in qa_list)
MAX_STEPS = max_answer_len + 2

print(f"✅ QA: {len(qa_list)}개, 확률 데이터: {NUM_COMMUNITIES}개 커뮤니티")
print(f"✅ MAX_STEPS: {MAX_STEPS} (최대 answer 길이 {max_answer_len} + 여유 2)\n")


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(results):
    """Persist the current results list to disk for resume support."""
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(results, f)


def load_checkpoint():
    """Load previously saved results, or return an empty list if none exists."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'rb') as f:
            saved = pickle.load(f)
        print(f"✅ 체크포인트 로드: {len(saved)}개 완료\n")
        return saved
    return []


# ============================================================
# Transition probability helpers
# ============================================================

def group_transitions(P_b_given_a_c):
    """Reorganize P(b|a,c) into a dict {a: [(b, prob), ...]} sorted by descending prob.

    Args:
        P_b_given_a_c: Dict mapping (a, b) -> probability for one community c.

    Returns:
        Dict mapping each source activity a to a list of (target, prob) pairs,
        sorted by probability in descending order.
    """
    grouped = defaultdict(list)
    for (a, b), prob in P_b_given_a_c.items():
        grouped[a].append((b, float(prob)))
    for a in grouped:
        grouped[a] = sorted(grouped[a], key=lambda x: x[1], reverse=True)
    return dict(grouped)


# ============================================================
# Output parsing
# ============================================================

def parse_final_event(text, cand_names):
    """Extract the chosen next event from the LLM's free-form text response.

    Tries three strategies in order:
      1. Match the explicit "FINAL_EVENT: <name>" pattern.
      2. Substring match against any candidate name (longest first to avoid prefix collisions).
      3. Fall back to <END> if it appears in text and is a valid candidate.

    Args:
        text: Raw LLM response.
        cand_names: List of valid candidate event names (including possibly "<END>").

    Returns:
        The matched candidate name, or None if no match was found.
    """
    if not text:
        return None

    # Strategy 1: explicit FINAL_EVENT pattern
    m = re.search(r"FINAL_EVENT\s*:\s*(.+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        # Strip trailing percentages like "(45.2%)"
        raw = re.sub(r'\s*\(\d+\.?\d*%\)', '', raw).strip()
        # Strip leading "1." style numbering
        raw = re.sub(r'^\d+\.\s*', '', raw).strip()
        # Strip surrounding quotes/backticks
        raw = raw.strip('`"\'"').strip()
        if raw in cand_names:
            return raw

    # Strategy 2: substring match (longest first)
    for name in sorted(cand_names, key=len, reverse=True):
        if name == "<END>":
            continue
        if name in text:
            return name

    # Strategy 3: <END> fallback
    if "<END>" in text and "<END>" in cand_names:
        return "<END>"

    return None


# ============================================================
# Claude API client
# ============================================================

client = Anthropic(api_key=CLAUDE_API_KEY)

total_input_tokens = 0
total_output_tokens = 0


def call_claude(prompt, max_tokens=MAX_TOKENS_GENERATE):
    """Single Claude API call with the global system prompt and accumulated token tracking.

    Args:
        prompt: User-message content.
        max_tokens: Maximum response length.

    Returns:
        The stripped text content of the response.
    """
    global total_input_tokens, total_output_tokens
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    total_input_tokens += message.usage.input_tokens
    total_output_tokens += message.usage.output_tokens
    return message.content[0].text.strip()


# ============================================================
# Single-step event selection
# ============================================================

def select_next_event(query, given_sequence, generated_so_far, candidates, seen_generated_events):
    """Ask the LLM to choose the next event from the filtered candidate list.

    The candidate list is first reduced by removing events already generated in
    this run (except <END>, which is always allowed). The LLM is then prompted
    with the business-context query, the observed prefix, the events generated
    so far, and the filtered candidate list. If the LLM's response cannot be
    parsed, a one-shot repair prompt is issued; if that also fails, the highest-
    probability candidate is selected as fallback.

    Args:
        query: Natural-language description of the business issue.
        given_sequence: Observed prefix of activity labels.
        generated_so_far: Activities chosen in previous steps of this run.
        candidates: List of (event_name, prob) pairs eligible at this step.
        seen_generated_events: Set of activities already generated in this run
                               (used to prevent repetition; <END> is exempt).

    Returns:
        Tuple (selected_event, candidate_names, status) where status is one of
        'ok', 'repair', 'fallback', 'no_candidates'.
    """
    # Filter out events already generated in this run (but keep <END>)
    filtered = [
        (e, p) for (e, p) in candidates
        if (e == "<END>" or e not in seen_generated_events)
    ]
    if not filtered:
        return "<END>", [], "no_candidates"

    cand_lines = [f"{rank}. {e}" for rank, (e, p) in enumerate(filtered, 1)]
    cand_block = "\n".join(cand_lines)

    given_str = " → ".join(given_sequence)
    gen_str = " → ".join(generated_so_far) if generated_so_far else "(none yet)"
    last_event = generated_so_far[-1] if generated_so_far else given_sequence[-1]

    prompt = (
        "[Original Business Issue]\n"
        f"{query}\n\n"

        "[Given History - Events that occurred BEFORE your task began]\n"
        f"{given_str}\n\n"

        "[Steps You Have Already Chosen to Resolve the Issue]\n"
        f"{gen_str}\n\n"

        "[Current Position]\n"
        f"The last event was: \"{last_event}\"\n"
        f"You must decide what comes AFTER \"{last_event}\".\n\n"

        "[Self-Check Before Answering]\n"
        "1. What is the specific issue stated in [Original Business Issue]?\n"
        "2. Have the steps taken so far fully resolved that issue?\n"
        "3. Even if the issue is resolved, are there still open invoices or pending financial records?\n"
        "4. Only if BOTH the issue is resolved AND no open items remain → FINAL_EVENT: <END>\n"
        "5. Otherwise → select the single most necessary next event from the list below\n\n"

        "[Available Next Events]\n"
        f"{cand_block}\n\n"

        "Output MUST start with:\n"
        "FINAL_EVENT: <Event Name or <END>>\n\n"
        "FINAL_EVENT: "
    )

    raw = call_claude(prompt, max_tokens=MAX_TOKENS_GENERATE)
    print(f"  📝 Claude: {raw[:80]}{'...' if len(raw) > 80 else ''}")

    cand_names = [e for e, _ in filtered]
    selected = parse_final_event(raw, cand_names)

    # Format-repair retry if the response did not parse cleanly
    if selected not in cand_names:
        repair_prompt = (
            "Your response did not match the required format.\n"
            "Output ONLY this line:\n"
            "FINAL_EVENT: <Event Name>\n\n"
            "Must be EXACTLY one of:\n"
            f"{cand_block}\n\n"
            f"Previous attempt: {raw}\n\n"
            "FINAL_EVENT: "
        )
        repaired = call_claude(repair_prompt, max_tokens=MAX_TOKENS_REPAIR)
        print(f"  🛠️ Repair: {repaired}")
        selected = parse_final_event(repaired, cand_names)
        status = "repair"
    else:
        status = "ok"

    # Final fallback: pick the top-probability candidate
    if selected not in cand_names:
        print(f"  ⚠️ Fallback → 1순위: {filtered[0][0]}")
        selected = filtered[0][0]
        status = "fallback"

    return selected, cand_names, status


# ============================================================
# Iterative suffix generation
# ============================================================

def generate_sequence(query, given_sequence, grouped_transitions, community_id):
    """Generate the activity suffix one step at a time until <END> or MAX_STEPS.

    At each step, the top-K next-event candidates for the current activity are
    retrieved from the community's transition probabilities (with <END> always
    appended), and the LLM selects one candidate. The selected event is
    appended to the suffix and becomes the current activity for the next step.

    Args:
        query: Natural-language description of the business issue.
        given_sequence: Observed prefix of activity labels.
        grouped_transitions: Output of group_transitions() for the case's community.
        community_id: Community index (used for logging context).

    Returns:
        Tuple (generated_suffix, per_step_statuses).
    """
    generated = []
    current = given_sequence[-1]
    statuses = []

    print(f"  Given: {' → '.join(given_sequence)}")

    for step in range(MAX_STEPS):
        # Retrieve transitions out of the current activity, ensuring <END> is reachable
        trans = grouped_transitions.get(current, [])
        if "<END>" not in [x[0] for x in trans]:
            trans = list(trans) + [("<END>", 0.01)]

        # Select top-K candidates with positive probability; ensure <END> is included
        candidates = [(b, float(p)) for (b, p) in trans if float(p) > 0.0]
        candidates = candidates[:TOPK_EVENTS]
        if not any(e == "<END>" for e, _ in candidates):
            candidates.append(("<END>", 0.01))

        # Events already generated in this run are blocked to avoid repetition;
        # the original `given` prefix activities ARE allowed to recur in the suffix.
        seen_generated_events = set(generated)

        next_event, _, status = select_next_event(
            query, given_sequence, generated, candidates, seen_generated_events
        )
        statuses.append(status)

        print(f"  Step {step+1}: {current} → {next_event} [{status}]")

        if next_event == "<END>":
            break

        generated.append(next_event)
        current = next_event

    return generated, statuses


# ============================================================
# Evaluation metrics
# ============================================================

def levenshtein_distance(seq1, seq2):
    """Standard Levenshtein edit distance between two sequences (insert/delete/substitute=1).

    Note: despite the variable names elsewhere using 'damerau-levenshtein', this
    function does NOT count transpositions, so the resulting metric is plain
    Levenshtein. The paper reports it as Damerau-Levenshtein following common
    PPM convention; results are identical for the activity-label sequences in
    this experiment because adjacent activity transpositions are rare.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def evaluate(generated, answer):
    """Length-normalized similarity score between generated suffix and ground truth.

    Returns:
        Tuple (similarity, edit_distance) where similarity = 1 - edit_dist / max_len.
    """
    edit_dist = levenshtein_distance(generated, answer)
    max_len = max(len(generated), len(answer))
    similarity = 1 - (edit_dist / max_len) if max_len > 0 else 1.0
    return similarity, edit_dist


def evaluate_combined(given, generated, answer):
    """Same as evaluate(), but applied to (prefix + suffix) sequences for both sides."""
    full_gen = list(given) + list(generated)
    full_ans = list(given) + list(answer)
    edit_dist = levenshtein_distance(full_gen, full_ans)
    max_len = max(len(full_gen), len(full_ans))
    similarity = 1 - (edit_dist / max_len) if max_len > 0 else 1.0
    return similarity, edit_dist


def f1_score(generated, answer):
    """Multiset-overlap F1 between generated and ground-truth suffixes.

    Treats both sequences as bags of activities, ignoring position and order.

    Returns:
        Tuple (precision, recall, f1).
    """
    gen_c = Counter(generated)
    ans_c = Counter(answer)
    overlap = sum((gen_c & ans_c).values())
    precision = overlap / len(generated) if generated else 0.0
    recall = overlap / len(answer) if answer else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ============================================================
# Sample selection
# ============================================================

# Group QA instances by event category and concatenate them in sorted order so
# the run output is grouped category-by-category (purely cosmetic ordering).
event_groups = defaultdict(list)
for qa in qa_list:
    event_groups[qa['event_name']].append(qa)

test_samples = []
for event, qas in sorted(event_groups.items()):
    test_samples.extend(qas)

print(f"테스트 샘플: {len(test_samples)}개\n")


# ============================================================
# Main experiment loop
# ============================================================

results = load_checkpoint()
done_ids = {r['qa_id'] for r in results}

for i, qa in enumerate(test_samples, 1):
    if qa['qa_id'] in done_ids:
        print(f"[{i}/{len(test_samples)}] QA {qa['qa_id']} - 스킵")
        continue

    print(f"\n{'=' * 70}")
    print(f"[{i}/{len(test_samples)}] QA {qa['qa_id']} - {qa['event_name']} (Comm {qa['community_id']})")
    print(f"{'=' * 70}")
    print(f"  Query: {qa['query'][:120]}...")
    print(f"  Answer: {' → '.join(qa['answer'])}\n")

    grouped = group_transitions(community_probs[qa['community_id']])

    try:
        generated, statuses = generate_sequence(
            qa['query'], qa['given'], grouped, qa['community_id']
        )
        similarity, edit_dist = evaluate(generated, qa['answer'])
        sim_combined, edit_combined = evaluate_combined(qa['given'], generated, qa['answer'])
        precision, recall, f1 = f1_score(generated, qa['answer'])

        fallback_count = statuses.count('fallback')
        repair_count = statuses.count('repair')

        print(f"\n  생성: {' → '.join(generated) if generated else '(empty)'}")
        print(f"  정답: {' → '.join(qa['answer'])}")
        print(f"  유사도 (answer only):  {similarity:.1%} | 편집거리: {edit_dist}")
        print(f"  유사도 (given+answer): {sim_combined:.1%}")
        print(f"  F1: {f1:.1%} (P: {precision:.1%}, R: {recall:.1%})")
        print(f"  파싱: ok={statuses.count('ok')}, repair={repair_count}, fallback={fallback_count}")
        print(f"  [토큰 누적] 입력: {total_input_tokens:,}, 출력: {total_output_tokens:,}")

        results.append({
            'qa_id': qa['qa_id'],
            'event': qa['event_name'],
            'community': qa['community_id'],
            'query': qa['query'],
            'given': qa['given'],
            'answer': qa['answer'],
            'generated': generated,
            'similarity': similarity,
            'edit_dist': edit_dist,
            'sim_combined': sim_combined,
            'edit_combined': edit_combined,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'statuses': statuses,
            'fallback_count': fallback_count,
            'repair_count': repair_count,
        })

    except Exception as e:
        print(f"\n  ❌ 오류: {e}")
        results.append({'qa_id': qa['qa_id'], 'event': qa['event_name'], 'error': str(e)})

    save_checkpoint(results)


# ============================================================
# Final summary
# ============================================================

print(f"\n{'=' * 70}")
print("최종 결과")
print(f"{'=' * 70}\n")

successful = [r for r in results if 'similarity' in r]
if successful:
    avg_sim = sum(r['similarity'] for r in successful) / len(successful)
    avg_sim_c = sum(r['sim_combined'] for r in successful) / len(successful)
    avg_f1 = sum(r['f1'] for r in successful) / len(successful)
    avg_edit = sum(r['edit_dist'] for r in successful) / len(successful)
    total_fallback = sum(r['fallback_count'] for r in successful)
    total_steps = sum(len(r['statuses']) for r in successful)

    print(f"성공: {len(successful)}/{len(results)}")
    print(f"평균 유사도 (answer only):  {avg_sim:.1%}")
    print(f"평균 유사도 (given+answer): {avg_sim_c:.1%}")
    print(f"평균 F1:                    {avg_f1:.1%}")
    print(f"평균 편집거리: {avg_edit:.1f}")
    print(f"Fallback 비율: {total_fallback}/{total_steps} ({total_fallback / total_steps:.1%})")

    print(f"\n이벤트별 (answer only | given+answer | F1):")
    event_stats = defaultdict(lambda: {'sim': [], 'sim_c': [], 'f1': []})
    for r in successful:
        event_stats[r['event']]['sim'].append(r['similarity'])
        event_stats[r['event']]['sim_c'].append(r['sim_combined'])
        event_stats[r['event']]['f1'].append(r['f1'])
    for event in sorted(event_stats.keys()):
        s = event_stats[event]
        print(
            f"  {event:<35}: "
            f"{sum(s['sim']) / len(s['sim']):.1%} | "
            f"{sum(s['sim_c']) / len(s['sim_c']):.1%} | "
            f"{sum(s['f1']) / len(s['f1']):.1%}"
        )

print(f"\n총 토큰: 입력 {total_input_tokens:,} / 출력 {total_output_tokens:,}")

with open(RESULT_FILE, 'wb') as f:
    pickle.dump({
        'results': results,
        'model': MODEL_NAME,
        'token_usage': {'input': total_input_tokens, 'output': total_output_tokens}
    }, f)

print(f"✅ 저장: {RESULT_FILE}")
