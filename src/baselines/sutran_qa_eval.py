"""
SuTraN QA 평가 스크립트
- best model (epoch 68) 로드
- QA 180개 더미 타임스탬프로 평가
"""

import os, sys, pickle
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import torch

SUTRAN_DIR  = "/workspace/hojun/SuffixTransformerNetwork"
sys.path.insert(0, SUTRAN_DIR)

from SuTraN.SuTraN import SuTraN
from SuTraN.inference_procedure import inference_loop

LOG_NAME   = "BPIC_19"
LOG_DIR    = os.path.join(SUTRAN_DIR, LOG_NAME)
MODEL_PATH = os.path.join(LOG_DIR, "SUTRAN_DA_results", "model_epoch_68.pt")
QA_PATH    = "/workspace/hojun/qa_dataset_final.pkl"
OUT_PATH   = os.path.join(LOG_DIR, "sutran_qa_results.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {device}")

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

cardin_dict         = load_dict(os.path.join(LOG_DIR, f'{LOG_NAME}_cardin_dict.pkl'))
cardin_list_prefix  = load_dict(os.path.join(LOG_DIR, f'{LOG_NAME}_cardin_list_prefix.pkl'))
cardin_list_suffix  = load_dict(os.path.join(LOG_DIR, f'{LOG_NAME}_cardin_list_suffix.pkl'))
num_cols_dict       = load_dict(os.path.join(LOG_DIR, f'{LOG_NAME}_num_cols_dict.pkl'))
cat_cols_dict       = load_dict(os.path.join(LOG_DIR, f'{LOG_NAME}_cat_cols_dict.pkl'))
categ_mapping       = load_dict(os.path.join(LOG_DIR, f'{LOG_NAME}_categ_mapping.pkl'))
train_means_dict    = load_dict(os.path.join(LOG_DIR, f'{LOG_NAME}_train_means_dict.pkl'))
train_std_dict      = load_dict(os.path.join(LOG_DIR, f'{LOG_NAME}_train_std_dict.pkl'))

num_activities      = cardin_dict['concept:name'] + 2
num_numericals_pref = len(num_cols_dict['prefix_df'])
num_numericals_suf  = len(num_cols_dict['suffix_df'])
num_cat_pref        = len(cat_cols_dict['prefix_df'])
num_cols_pref       = num_cols_dict['prefix_df']
tss_index           = num_cols_pref.index('ts_start')

train_data  = torch.load(os.path.join(LOG_DIR, 'train_tensordataset.pt'), map_location='cpu')
window_size = train_data[0].shape[1]
print(f"num_activities={num_activities}  window_size={window_size}  num_cat_pref={num_cat_pref}")

mean_std_ttne = [train_means_dict['timeLabel_df'][0], train_std_dict['timeLabel_df'][0]]
mean_std_tsp  = [train_means_dict['suffix_df'][0],    train_std_dict['suffix_df'][0]]
mean_std_tss  = [train_means_dict['suffix_df'][1],    train_std_dict['suffix_df'][1]]
mean_std_rrt  = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]

# 모델 로드
model = SuTraN(
    num_activities                = num_activities,
    d_model                       = 32,
    cardinality_categoricals_pref = cardin_list_prefix,
    num_numericals_pref           = num_numericals_pref,
    num_prefix_encoder_layers     = 4,
    num_decoder_layers            = 4,
    num_heads                     = 8,
    d_ff                          = 128,
    dropout                       = 0.2,
    remaining_runtime_head        = True,
    layernorm_embeds              = True,
    outcome_bool                  = False,
)
ck = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ck['model_state_dict'])
model.to(device)
model.eval()
print("모델 로드 완료")

act_label  = 'concept:name'
act_map    = categ_mapping[act_label]
idx_to_act = {v + 1: k for k, v in act_map.items()}
eos_idx    = num_activities - 1

with open(QA_PATH, 'rb') as f:
    qa_raw = pickle.load(f)
qa_dataset = qa_raw.get('qa_dataset', qa_raw) if isinstance(qa_raw, dict) else qa_raw
print(f"QA: {len(qa_dataset)}개")

base = datetime(2018, 6, 1, 9, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)

means_arr = np.array(train_means_dict['prefix_df'])
stds_arr  = np.array(train_std_dict['prefix_df'])

def build_tensors(given, answer):
    n = len(given)
    leftpad = window_size - n
    prefix_ts = [base + timedelta(hours=i) for i in range(n)]

    # cat tensors
    cat_tensors = []
    for col in cat_cols_dict['prefix_df']:
        t = torch.zeros(1, window_size, dtype=torch.long)
        if col == act_label:
            for i, act in enumerate(given):
                t[0, i + leftpad] = act_map.get(act, max(act_map.values())) + 1
        cat_tensors.append(t)

    # numeric tensor
    case_start = prefix_ts[0]; last_t = prefix_ts[0]
    num_t = torch.zeros(1, window_size, num_numericals_pref, dtype=torch.float32)
    for i, ts in enumerate(prefix_ts):
        dp = 0.0 if i == 0 else (ts - last_t).total_seconds()
        ds = (ts - case_start).total_seconds()
        last_t = ts
        for j, col in enumerate(num_cols_pref):
            val = ds if col == 'ts_start' else (dp if col == 'ts_prev' else 0.0)
            std = stds_arr[j] if stds_arr[j] != 0 else 1.0
            num_t[0, i + leftpad, j] = (val - means_arr[j]) / std

    # pad mask
    pad_mask = torch.ones(1, window_size, dtype=torch.bool)
    pad_mask[0, leftpad:] = False

    # suffix labels
    act_label_t = torch.zeros(1, window_size, dtype=torch.long)
    for i, act in enumerate(answer[:window_size-1]):
        act_label_t[0, i] = act_map.get(act, max(act_map.values())) + 1
    if len(answer) < window_size:
        act_label_t[0, len(answer)] = eos_idx

    ttne_label = torch.zeros(1, window_size, 1, dtype=torch.float32)
    rrt_label  = torch.zeros(1, window_size, 1, dtype=torch.float32)
    suf_act    = torch.zeros(1, window_size, dtype=torch.long)
    suf_num    = torch.zeros(1, window_size, num_numericals_suf, dtype=torch.float32)

    return cat_tensors, num_t, pad_mask, suf_act, suf_num, ttne_label, rrt_label, act_label_t

# 배치 구성
print("텐서 구성 중...")
qa_ids  = list(qa_dataset.keys())
all_cat = [[] for _ in range(num_cat_pref)]
all_num, all_pad, all_suf_act, all_suf_num = [], [], [], []
all_ttne, all_rrt, all_act_lbl = [], [], []

for qa_id in qa_ids:
    qa     = qa_dataset[qa_id]
    given  = list(qa['given'])
    answer = list(qa['answer'])
    cats, num_t, pad, suf_act, suf_num, ttne, rrt, act_lbl = build_tensors(given, answer)
    for i, ct in enumerate(cats):
        all_cat[i].append(ct)
    all_num.append(num_t); all_pad.append(pad)
    all_suf_act.append(suf_act); all_suf_num.append(suf_num)
    all_ttne.append(ttne); all_rrt.append(rrt); all_act_lbl.append(act_lbl)

batch_cat     = [torch.cat(all_cat[i], dim=0) for i in range(num_cat_pref)]
batch_num     = torch.cat(all_num, dim=0)
batch_pad     = torch.cat(all_pad, dim=0)
batch_suf_act = torch.cat(all_suf_act, dim=0)
batch_suf_num = torch.cat(all_suf_num, dim=0)
batch_ttne    = torch.cat(all_ttne, dim=0)
batch_rrt     = torch.cat(all_rrt, dim=0)
batch_act_lbl = torch.cat(all_act_lbl, dim=0)

inference_dataset = tuple(batch_cat) + (batch_num, batch_pad,
                                         batch_suf_act, batch_suf_num,
                                         batch_ttne, batch_rrt, batch_act_lbl)
print(f"배치: {len(qa_ids)}개  텐서 수: {len(inference_dataset)}")

# inference
print("\nSuTraN inference 시작...")
results_list = inference_loop(
    model                  = model,
    inference_dataset      = inference_dataset,
    remaining_runtime_head = True,
    outcome_bool           = False,
    num_categoricals_pref  = num_cat_pref,
    mean_std_ttne          = mean_std_ttne,
    mean_std_tsp           = mean_std_tsp,
    mean_std_tss           = mean_std_tss,
    mean_std_rrt           = mean_std_rrt,
    val_batch_size         = 180,
)
print(f"inference_loop DL: {results_list[2]:.4f}")

# 예측 suffix 추출 (F1용)
print("예측 suffix 추출 중...")
model.eval()
with torch.no_grad():
    inputs = [t.to(device) for t in tuple(batch_cat) + (batch_num, batch_pad,
                                                          batch_suf_act, batch_suf_num)]
    outputs = model(inputs, window_size, mean_std_ttne, mean_std_tsp, mean_std_tss)
suffix_acts = outputs[0].cpu()  # (N, W)

def dl_similarity(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or  not s2: return 0.0
    l1, l2 = len(s1), len(s2)
    d = np.arange(l2+1, dtype=float)
    for i in range(1, l1+1):
        prev=d.copy(); d[0]=i
        for j in range(1, l2+1):
            cost = 0 if s1[i-1]==s2[j-1] else 1
            d[j] = min(d[j-1]+1, prev[j]+1, prev[j-1]+cost)
            if i>1 and j>1 and s1[i-1]==s2[j-2] and s1[i-2]==s2[j-1]:
                d[j] = min(d[j], prev[j-1])
    return max(0.0, 1-d[l2]/max(l1,l2))

def f1_sets(pred, answer):
    pc, ac = Counter(pred), Counter(answer)
    tp = sum((pc & ac).values())
    if tp == 0: return 0.0
    p = tp/sum(pc.values()); r = tp/sum(ac.values())
    return 2*p*r/(p+r)

results = {}
dl_list, f1_list = [], []

for i, qa_id in enumerate(qa_ids):
    qa     = qa_dataset[qa_id]
    answer = list(qa['answer'])
    given  = list(qa['given'])
    predicted = []
    for idx in suffix_acts[i].tolist():
        if idx == eos_idx or idx == 0: break
        act = idx_to_act.get(int(idx), None)
        if act: predicted.append(act)

    sim = dl_similarity(predicted, answer)
    f1  = f1_sets(predicted, answer)
    dl_list.append(sim); f1_list.append(f1)
    results[qa_id] = {'given': given, 'answer': answer, 'predicted': predicted,
                      'dl_similarity': sim, 'f1': f1}

summary = {'mean_dl_similarity': float(np.mean(dl_list)),
           'mean_f1': float(np.mean(f1_list)), 'n': len(dl_list)}

print(f"\n{'='*50}")
print(f"SuTraN 결과 ({summary['n']}개)")
print(f"  DL Similarity : {summary['mean_dl_similarity']:.4f}")
print(f"  F1 Score      : {summary['mean_f1']:.4f}")
print(f"{'='*50}")

by_type = {}
for qa_id, r in results.items():
    key = qa_dataset[qa_id]['event_key']
    by_type.setdefault(key, {'f1':[], 'dl':[]})
    by_type[key]['f1'].append(r['f1']); by_type[key]['dl'].append(r['dl_similarity'])
print("\n이벤트 타입별:")
for k, v in sorted(by_type.items()):
    print(f"  {k:<35} N={len(v['f1']):3d}  DL={np.mean(v['dl']):.3f}  F1={np.mean(v['f1']):.3f}")

with open(OUT_PATH, 'wb') as f:
    pickle.dump({'results': results, 'summary': summary}, f)
print(f"\n결과 저장: {OUT_PATH}")
