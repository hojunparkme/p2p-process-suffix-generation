"""
Tax et al. (2017) LSTM Baseline - BPI 2019 (PyTorch 버전)
- GPU 자동 사용
- maxlen: 98.5th percentile
- 배치 generator로 메모리 효율적 학습
- QA 180개 평가: DL similarity + F1
"""

import os, sys, pickle, argparse
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

XES_PATH   = "/workspace/hojun/BPI_Challenge_2019.xes"
QA_PATH    = "/workspace/hojun/qa_dataset_final.pkl"
OUTPUT_DIR = "/workspace/hojun/tax_lstm_output"
EOS        = '!'

VALID_ACTS = {
    'Block Purchase Order Item', 'Cancel Goods Receipt', 'Cancel Invoice Receipt',
    'Cancel Subsequent Invoice', 'Change Approval for Purchase Order',
    'Change Delivery Indicator', 'Change Price', 'Change Quantity',
    'Change Storage Location', 'Clear Invoice', 'Create Purchase Order Item',
    'Create Purchase Requisition Item', 'Delete Purchase Order Item',
    'Reactivate Purchase Order Item', 'Receive Order Confirmation',
    'Record Goods Receipt', 'Record Invoice Receipt', 'Record Service Entry Sheet',
    'Record Subsequent Invoice', 'Release Purchase Order',
    'Release Purchase Requisition', 'Remove Payment Block',
    'Update Order Confirmation', 'Vendor creates debit memo', 'Vendor creates invoice',
}

# ─────────────────────────────────────────────
# 1. XES 파싱
# ─────────────────────────────────────────────
def parse_xes(path, valid_acts=None, max_cases=None):
    print(f"XES 파싱 중: {path}")
    cases = {}
    in_trace = in_event = False
    current_case_id = None
    current_events = []
    current_event = {}
    trace_count = 0

    for xml_event, elem in ET.iterparse(path, events=('start', 'end')):
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if xml_event == 'start':
            if tag == 'trace':
                in_trace = True
                current_case_id = None
                current_events = []
            elif tag == 'event' and in_trace:
                in_event = True
                current_event = {}

        elif xml_event == 'end':
            if tag in ('string', 'date', 'int', 'float', 'boolean') and in_trace:
                key   = elem.attrib.get('key', '')
                value = elem.attrib.get('value', '')
                if not in_event and key == 'concept:name':
                    current_case_id = value
                elif in_event:
                    if key == 'concept:name':
                        current_event['act'] = value
                    elif key == 'time:timestamp':
                        current_event['ts'] = value

            elif tag == 'event' and in_trace:
                in_event = False
                act    = current_event.get('act', '')
                ts_str = current_event.get('ts', '')
                if act and ts_str and (valid_acts is None or act in valid_acts):
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        current_events.append((act, ts))
                    except Exception:
                        pass

            elif tag == 'trace':
                in_trace = in_event = False
                if current_case_id and len(current_events) >= 2:
                    cases[current_case_id] = current_events
                trace_count += 1
                if trace_count % 20000 == 0:
                    print(f"  {trace_count:,} traces 처리...")
                if max_cases and trace_count >= max_cases:
                    break
            elem.clear()

    print(f"  완료: {len(cases):,}개 케이스")
    return cases


# ─────────────────────────────────────────────
# 2. Vocab
# ─────────────────────────────────────────────
def build_vocab(cases):
    all_acts = sorted({a for evts in cases.values() for a, _ in evts})
    char_idx     = {c: i for i, c in enumerate(all_acts)}
    target_chars = all_acts + [EOS]
    tgt_idx      = {c: i for i, c in enumerate(target_chars)}
    tgt_idx_char = {i: c for i, c in enumerate(target_chars)}
    print(f"  활동: {len(all_acts)}  target(+EOS): {len(target_chars)}")
    return char_idx, tgt_idx, tgt_idx_char


# ─────────────────────────────────────────────
# 3. 통계 & maxlen
# ─────────────────────────────────────────────
def compute_stats(cases, train_ids, percentile=98.5):
    lengths, diffs, diffs2 = [], [], []
    for cid in train_ids:
        evts = cases[cid]
        lengths.append(len(evts) + 1)
        ts_list = [t for _, t in evts]
        for i, ts in enumerate(ts_list):
            diffs.append(0 if i == 0 else (ts - ts_list[i-1]).total_seconds())
            diffs2.append((ts - ts_list[0]).total_seconds())

    maxlen   = int(np.percentile(lengths, percentile))
    divisor  = float(np.mean(diffs))  if diffs  else 1.0
    divisor2 = float(np.mean(diffs2)) if diffs2 else 1.0
    print(f"  길이 분포: min={min(lengths)} mean={np.mean(lengths):.1f} "
          f"p{percentile}={maxlen} max={max(lengths)}")
    print(f"  maxlen={maxlen}  divisor={divisor:.1f}  divisor2={divisor2:.1f}")
    return maxlen, divisor, divisor2


# ─────────────────────────────────────────────
# 4. Dataset
# ─────────────────────────────────────────────
def case_to_tensor(evts, maxlen, char_idx, tgt_idx, divisor, divisor2):
    """케이스 → (X, y_act, y_time) 텐서 리스트"""
    acts    = [a for a, _ in evts] + [EOS]
    ts_list = [t for _, t in evts]
    n_input   = len(char_idx)
    num_feats = n_input + 5
    n_target  = len(tgt_idx)

    t_prev, t_start, t_day, t_week = [], [], [], []
    case_start = ts_list[0]
    last_t = ts_list[0]
    for i, ts in enumerate(ts_list):
        t_prev.append(0 if i == 0 else (ts - last_t).total_seconds())
        t_start.append((ts - case_start).total_seconds())
        midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        t_day.append((ts - midnight).total_seconds())
        t_week.append(float(ts.weekday()))
        last_t = ts

    results = []
    for i in range(1, len(acts)):
        sent = acts[max(0, i-maxlen):i]   # truncate
        L    = len(sent)
        leftpad = maxlen - L

        x = np.zeros((maxlen, num_feats), dtype=np.float32)
        for t_idx, ch in enumerate(sent):
            orig = i - L + t_idx
            if ch in char_idx:
                x[t_idx + leftpad, char_idx[ch]] = 1
            _tp = t_prev[orig]  if orig < len(t_prev)  else 0
            _ts = t_start[orig] if orig < len(t_start) else 0
            _td = t_day[orig]   if orig < len(t_day)   else 0
            _tw = t_week[orig]  if orig < len(t_week)  else 0
            x[t_idx + leftpad, n_input]     = t_idx + 1
            x[t_idx + leftpad, n_input + 1] = _tp / (divisor + 1e-9)
            x[t_idx + leftpad, n_input + 2] = _ts / (divisor2 + 1e-9)
            x[t_idx + leftpad, n_input + 3] = _td / 86400.0
            x[t_idx + leftpad, n_input + 4] = _tw / 7.0

        nc = acts[i]
        ya = tgt_idx.get(nc, 0)
        yt = (t_prev[i] / (divisor + 1e-9)) if (nc != EOS and i < len(t_prev)) else 0.0

        results.append((x, ya, np.float32(yt)))
    return results


class BPI19Dataset(Dataset):
    def __init__(self, cases, case_ids, maxlen, char_idx, tgt_idx, divisor, divisor2):
        self.data = []
        print(f"  Dataset 구축 중 ({len(case_ids):,}개 케이스)...")
        for idx, cid in enumerate(case_ids):
            self.data.extend(
                case_to_tensor(cases[cid], maxlen, char_idx, tgt_idx, divisor, divisor2)
            )
            if (idx + 1) % 20000 == 0:
                print(f"    {idx+1:,}/{len(case_ids):,}  prefix={len(self.data):,}")
        print(f"  총 prefix: {len(self.data):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, ya, yt = self.data[idx]
        return (
            torch.FloatTensor(x),
            torch.tensor(ya, dtype=torch.long),
            torch.tensor(yt, dtype=torch.float),
        )


# ─────────────────────────────────────────────
# 5. 모델
# ─────────────────────────────────────────────
class TaxLSTM(nn.Module):
    def __init__(self, num_feats, n_target, hidden=100, dropout=0.2):
        super().__init__()
        self.lstm_shared = nn.LSTM(num_feats, hidden, batch_first=True, dropout=dropout)
        self.bn_shared   = nn.BatchNorm1d(hidden)
        self.lstm_act    = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)
        self.bn_act      = nn.BatchNorm1d(hidden)
        self.lstm_time   = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout)
        self.bn_time     = nn.BatchNorm1d(hidden)
        self.fc_act      = nn.Linear(hidden, n_target)
        self.fc_time     = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm_shared(x)                     # (B, T, H)
        out_bn = self.bn_shared(out[:, -1, :])           # (B, H)

        out_a, _ = self.lstm_act(out)
        out_a_bn = self.bn_act(out_a[:, -1, :])
        act_logits = self.fc_act(out_a_bn)               # (B, n_target)

        out_t, _ = self.lstm_time(out)
        out_t_bn = self.bn_time(out_t[:, -1, :])
        time_pred = self.fc_time(out_t_bn).squeeze(-1)   # (B,)

        return act_logits, time_pred


# ─────────────────────────────────────────────
# 6. 학습
# ─────────────────────────────────────────────
def train_model(cases, train_ids, val_ids, meta, model_dir,
                batch_size=512, epochs=500, patience=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n디바이스: {device}")

    maxlen   = meta['maxlen']
    char_idx = meta['char_idx']
    tgt_idx  = meta['tgt_idx']
    divisor  = meta['divisor']
    divisor2 = meta['divisor2']
    n_target = meta['n_target']
    num_feats= meta['num_feats']

    print("Train dataset 구축...")
    train_ds = BPI19Dataset(cases, train_ids, maxlen, char_idx, tgt_idx, divisor, divisor2)
    print("Val dataset 구축...")
    val_ds   = BPI19Dataset(cases, val_ids,   maxlen, char_idx, tgt_idx, divisor, divisor2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = TaxLSTM(num_feats, n_target).to(device)
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=0.002)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)
    ce_loss   = nn.CrossEntropyLoss()
    mae_loss  = nn.L1Loss()

    best_val_loss = float('inf')
    best_path     = os.path.join(model_dir, 'best_model.pt')
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X, ya, yt in train_loader:
            X, ya, yt = X.to(device), ya.to(device), yt.to(device)
            optimizer.zero_grad()
            logits, time_pred = model(X)
            loss = ce_loss(logits, ya) + mae_loss(time_pred, yt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_ds)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, ya, yt in val_loader:
                X, ya, yt = X.to(device), ya.to(device), yt.to(device)
                logits, time_pred = model(X)
                loss = ce_loss(logits, ya) + mae_loss(time_pred, yt)
                val_loss += loss.item() * len(X)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  → Best model saved (val={val_loss:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (patience={patience})")
                break

    print(f"Best model: {best_path}  (val_loss={best_val_loss:.4f})")
    return best_path


# ─────────────────────────────────────────────
# 7. Suffix Prediction
# ─────────────────────────────────────────────
def predict_suffix(model, prefix_events, meta, device, max_steps=25):
    maxlen       = meta['maxlen']
    num_feats    = meta['num_feats']
    n_input      = meta['n_input']
    char_idx     = meta['char_idx']
    tgt_idx_char = meta['tgt_idx_char']
    divisor      = meta['divisor']
    divisor2     = meta['divisor2']

    current   = list(prefix_events)
    predicted = []
    model.eval()

    with torch.no_grad():
        for _ in range(max_steps):
            window     = current[-maxlen:] if len(current) > maxlen else current
            leftpad    = maxlen - len(window)
            case_start = window[0][1]
            last_t     = window[0][1]

            x = np.zeros((1, maxlen, num_feats), dtype=np.float32)
            for idx, (act, ts) in enumerate(window):
                dp = 0 if idx == 0 else (ts - last_t).total_seconds()
                ds = (ts - case_start).total_seconds()
                midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
                dm = (ts - midnight).total_seconds()
                dw = float(ts.weekday())
                if act in char_idx:
                    x[0, idx + leftpad, char_idx[act]] = 1
                x[0, idx + leftpad, n_input]     = idx + 1
                x[0, idx + leftpad, n_input + 1] = dp / (divisor + 1e-9)
                x[0, idx + leftpad, n_input + 2] = ds / (divisor2 + 1e-9)
                x[0, idx + leftpad, n_input + 3] = dm / 86400.0
                x[0, idx + leftpad, n_input + 4] = dw / 7.0
                last_t = ts

            X_t = torch.FloatTensor(x).to(device)
            logits, time_pred = model(X_t)
            next_act = tgt_idx_char[int(logits[0].argmax())]

            if next_act == EOS:
                break

            predicted.append(next_act)
            dt = max(0.0, float(time_pred[0].item()) * (divisor + 1e-9))
            current.append((next_act, current[-1][1] + timedelta(seconds=dt)))

    return predicted


# ─────────────────────────────────────────────
# 8. 평가 지표
# ─────────────────────────────────────────────
def dl_similarity(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or  not s2: return 0.0
    l1, l2 = len(s1), len(s2)
    d = np.arange(l2 + 1, dtype=float)
    for i in range(1, l1 + 1):
        prev = d.copy(); d[0] = i
        for j in range(1, l2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            d[j] = min(d[j-1]+1, prev[j]+1, prev[j-1]+cost)
            if i>1 and j>1 and s1[i-1]==s2[j-2] and s1[i-2]==s2[j-1]:
                d[j] = min(d[j], prev[j-1])
    return max(0.0, 1 - d[l2] / max(l1, l2))

def f1_sets(pred, answer):
    pc, ac = Counter(pred), Counter(answer)
    tp = sum((pc & ac).values())
    if tp == 0: return 0.0
    p = tp / sum(pc.values())
    r = tp / sum(ac.values())
    return 2*p*r/(p+r)


# ─────────────────────────────────────────────
# 9. QA 평가
# ─────────────────────────────────────────────
def evaluate_on_qa(model, qa_dataset, meta, cases, device):
    results = {}
    dl_list, f1_list = [], []
    print("\nQA 평가 시작...")

    for qa_id, qa in qa_dataset.items():
        given  = list(qa['given'])
        answer = list(qa['answer'])

        # prefix_events 찾기 (타임스탬프 포함)
        prefix_events = None
        for cid, evts in cases.items():
            acts = [a for a, _ in evts]
            for start in range(len(acts) - len(given) + 1):
                if acts[start:start+len(given)] == given:
                    prefix_events = evts[start:start+len(given)]
                    break
            if prefix_events:
                break

        if prefix_events is None:
            base = datetime(2018, 6, 1, 9, 0, 0,
                            tzinfo=datetime.now().astimezone().tzinfo)
            prefix_events = [(act, base + timedelta(hours=i))
                             for i, act in enumerate(given)]

        try:
            predicted = predict_suffix(model, prefix_events, meta, device)
        except Exception as e:
            print(f"  QA {qa_id} 오류: {e}")
            predicted = []

        sim = dl_similarity(predicted, answer)
        f1  = f1_sets(predicted, answer)
        dl_list.append(sim)
        f1_list.append(f1)

        results[qa_id] = {
            'given': given, 'answer': answer, 'predicted': predicted,
            'dl_similarity': sim, 'f1': f1,
        }
        if len(results) % 30 == 0:
            print(f"  {len(results)}/{len(qa_dataset)}  "
                  f"DL={np.mean(dl_list):.3f}  F1={np.mean(f1_list):.3f}")

    summary = {
        'mean_dl_similarity': float(np.mean(dl_list)),
        'mean_f1': float(np.mean(f1_list)),
        'n': len(dl_list),
    }

    print(f"\n{'='*50}")
    print(f"Tax et al. LSTM 결과 ({summary['n']}개)")
    print(f"  DL Similarity : {summary['mean_dl_similarity']:.4f}")
    print(f"  F1 Score      : {summary['mean_f1']:.4f}")
    print(f"{'='*50}")

    # 이벤트 타입별
    by_type = {}
    for qa_id, r in results.items():
        key = qa_dataset[qa_id]['event_key']
        by_type.setdefault(key, {'f1': [], 'dl': []})
        by_type[key]['f1'].append(r['f1'])
        by_type[key]['dl'].append(r['dl_similarity'])
    print("\n이벤트 타입별:")
    for k, v in sorted(by_type.items()):
        print(f"  {k:<35} N={len(v['f1']):3d}  "
              f"DL={np.mean(v['dl']):.3f}  F1={np.mean(v['f1']):.3f}")

    return results, summary


# ─────────────────────────────────────────────
# 10. 메인
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xes',        default=XES_PATH)
    parser.add_argument('--qa',         default=QA_PATH)
    parser.add_argument('--output',     default=OUTPUT_DIR)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--max_cases',  type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--percentile', type=float, default=98.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output, exist_ok=True)
    model_dir   = os.path.join(args.output, 'models')
    meta_path   = os.path.join(args.output, 'meta.pkl')
    result_path = os.path.join(args.output, 'tax_lstm_results.pkl')
    os.makedirs(model_dir, exist_ok=True)

    # 파싱
    cases = parse_xes(args.xes, valid_acts=VALID_ACTS, max_cases=args.max_cases)

    # vocab
    char_idx, tgt_idx, tgt_idx_char = build_vocab(cases)

    # 분리
    case_ids  = sorted(cases.keys())
    n         = len(case_ids)
    train_ids = case_ids[:int(n * 2/3)]
    val_ids   = case_ids[int(n * 2/3):int(n * 5/6)]
    test_ids  = case_ids[int(n * 5/6):]
    print(f"학습: {len(train_ids):,}  val: {len(val_ids):,}  test: {len(test_ids):,}")

    # 통계
    maxlen, divisor, divisor2 = compute_stats(cases, train_ids, args.percentile)

    n_input   = len(char_idx)
    n_target  = len(tgt_idx)
    num_feats = n_input + 5

    meta = dict(
        maxlen=maxlen, n_input=n_input, n_target=n_target, num_feats=num_feats,
        char_idx=char_idx, tgt_idx=tgt_idx, tgt_idx_char=tgt_idx_char,
        divisor=divisor, divisor2=divisor2,
    )
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Meta 저장: {meta_path}")

    best_path = os.path.join(model_dir, 'best_model.pt')

    if not args.skip_train:
        best_path = train_model(
            cases, train_ids, val_ids, meta, model_dir, args.batch_size
        )
    else:
        print(f"학습 건너뜀: {best_path}")

    # 모델 로드
    model = TaxLSTM(num_feats, n_target).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    print("모델 로드 완료")

    # QA 로드
    with open(args.qa, 'rb') as f:
        qa_raw = pickle.load(f)
    qa_dataset = qa_raw.get('qa_dataset', qa_raw) if isinstance(qa_raw, dict) else qa_raw
    print(f"QA: {len(qa_dataset)}개")

    # 평가
    results, summary = evaluate_on_qa(model, qa_dataset, meta, cases, device)

    with open(result_path, 'wb') as f:
        pickle.dump({'results': results, 'summary': summary, 'meta': meta}, f)
    print(f"결과 저장: {result_path}")


if __name__ == '__main__':
    main()
