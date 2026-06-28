# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import esm
import os
import random
import re
import warnings
import ssl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

# ================== 配置 ==================
DATA_DIR = './'
TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'GFP_data.xlsx')
WT_SEQ_FILE = os.path.join(DATA_DIR, 'AAseqs_of_4_GFP_proteins.txt')
EXCLUSION_FILE = os.path.join(DATA_DIR, 'Exclusion_List.csv')

ESM_MODEL_NAME = "esm2_t30_150M_UR50D"
N_CANDIDATES_TO_GENERATE = 2000
TOP_N_SELECT = 6
SEED = 42

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"使用设备: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================== 1. 加载 WT ==================
print("\n[1/5] 加载 WT 序列...")
def load_sequence(file_path, name):
    with open(file_path, 'r') as f:
        header, seq_lines = "", []
        for line in f:
            if line.startswith('>'):
                if name in header and seq_lines:
                    seq = "".join(seq_lines).strip()
                    seq = re.sub(r'[^A-Za-z]', '', seq).upper()
                    return seq
                header, seq_lines = line.strip(), []
            else:
                seq_lines.append(line.strip())
        if name in header and seq_lines:
            seq = "".join(seq_lines).strip()
            seq = re.sub(r'[^A-Za-z]', '', seq).upper()
            return seq
    return None

sfGFP_WT = load_sequence(WT_SEQ_FILE, "sfGFP")
avGFP_WT = load_sequence(WT_SEQ_FILE, "avGFP")

if sfGFP_WT is None or avGFP_WT is None:
    raise ValueError("无法读取 WT 序列，请检查文件。")

print(f"sfGFP 长度: {len(sfGFP_WT)}")
print(f"avGFP 长度: {len(avGFP_WT)}")

# ================== 2. 加载训练数据 ==================
print("\n[2/5] 加载训练数据...")
gfp_df = pd.read_excel(TRAIN_DATA_FILE, sheet_name='brightness')
avGFP_train_df = gfp_df[gfp_df['GFP type'] == 'avGFP'].copy()
if len(avGFP_train_df) < 100:
    print("avGFP 数据量不足，改用全部 GFP 类型数据。")
    avGFP_train_df = gfp_df.copy()

def generate_mutated_sequence(mut_str, wt_seq):
    if not isinstance(mut_str, str):
        return wt_seq
    mut_str = mut_str.strip().upper()
    if mut_str == 'WT':
        return wt_seq
    
    seq_list = list(wt_seq)
    parts = [p.strip() for p in mut_str.split(':') if p.strip()]
    
    for m in parts:
        match = re.match(r'([A-Z])(\d+)([A-Z*.]?)$', m, re.IGNORECASE)
        if match:
            orig, pos, new = match.groups()
            pos = int(pos) - 1
            if 0 <= pos < len(seq_list):
                if new == '*':
                    return None
                new_up = new.upper()
                if new_up not in 'ACDEFGHIKLMNPQRSTVWY':
                    continue
                seq_list[pos] = new_up
    return "".join(seq_list)

avGFP_train_df['full_sequence'] = avGFP_train_df['aaMutations'].apply(
    lambda x: generate_mutated_sequence(x, avGFP_WT)
)

avGFP_train_df.dropna(subset=['full_sequence'], inplace=True)
avGFP_train_df['Brightness'] = pd.to_numeric(avGFP_train_df['Brightness'], errors='coerce')
avGFP_train_df.dropna(subset=['Brightness'], inplace=True)

print(f"初始有效行数: {len(avGFP_train_df)}")

avGFP_train_df['full_sequence'] = avGFP_train_df['full_sequence'].str.upper()
avGFP_train_df = avGFP_train_df[avGFP_train_df['full_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', na=False)]
print(f"清理后有效行数: {len(avGFP_train_df)}")

if len(avGFP_train_df) < 50:
    print("⚠️ 有效训练数据不足，生成虚拟训练集（仅用于跑通流程）。")
    dummy_seqs = []
    dummy_vals = []
    for _ in range(1000):
        seq_list = list(avGFP_WT)
        num_mut = random.randint(1, 8)
        positions = random.sample(range(len(avGFP_WT)), num_mut)
        for p in positions:
            new = random.choice([aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != avGFP_WT[p]])
            seq_list[p] = new
        dummy_seqs.append("".join(seq_list))
        dummy_vals.append(random.uniform(1.0, 4.0))
    avGFP_train_df = pd.DataFrame({'full_sequence': dummy_seqs, 'Brightness': dummy_vals})
    print(f"生成虚拟数据 {len(avGFP_train_df)} 条。")

sampled_df = avGFP_train_df.sample(n=min(5000, len(avGFP_train_df)), random_state=SEED)
print(f"实际训练集大小: {len(sampled_df)}")

# ================== 3. 训练模型 ==================
print("\n[3/5] 生成 ESM 嵌入并训练随机森林...")
esm_model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
esm_model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()

def get_embeddings(seqs, model, converter, device, bs=16):
    embeds = []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            batch = seqs[i:i+bs]
            labels = [f"s{j}" for j in range(len(batch))]
            _, _, tokens = converter(list(zip(labels, batch)))
            tokens = tokens.to(device)
            results = model(tokens, repr_layers=[model.num_layers])
            reps = results["representations"][model.num_layers]
            for j, s in enumerate(batch):
                embeds.append(reps[j, 1:len(s)+1, :].mean(dim=0).cpu())
    if not embeds:
        return np.array([])
    return torch.stack(embeds).numpy()

X = get_embeddings(sampled_df['full_sequence'].tolist(), esm_model, batch_converter, DEVICE)
if X.shape[0] == 0:
    raise ValueError("嵌入失败，无有效数据。")
y = sampled_df['Brightness'].values
print(f"嵌入形状: {X.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED, max_depth=20)
rf.fit(X_train, y_train)
print(f"验证集 R²: {r2_score(y_val, rf.predict(X_val)):.4f}")

del esm_model, alphabet, batch_converter
torch.cuda.empty_cache()

# ================== 4. 生成候选 ==================
print("\n[4/5] 生成候选序列...")
protected = [64, 65, 66, 67, 153, 163]
protected_0 = [p-1 for p in protected]
all_pos = list(range(len(sfGFP_WT)))
mutable_pos = [p for p in all_pos if p not in protected_0]
aa_list = 'ACDEFGHIKLMNPQRSTVWY'

def generate_candidate():
    num_mut = random.randint(5, 20)
    if num_mut > len(mutable_pos):
        num_mut = len(mutable_pos)
    chosen = random.sample(mutable_pos, num_mut)
    seq = list(sfGFP_WT)
    muts = []
    for p in chosen:
        orig = sfGFP_WT[p]
        new = random.choice([a for a in aa_list if a != orig])
        seq[p] = new
        muts.append(f"{orig}{p+1}{new}")
    return "".join(seq), ":".join(sorted(muts, key=lambda x: int(re.search(r'\d+', x).group())))

candidate_dict = {}
attempts = 0
while len(candidate_dict) < N_CANDIDATES_TO_GENERATE and attempts < N_CANDIDATES_TO_GENERATE * 10:
    attempts += 1
    s, m = generate_candidate()
    if s not in candidate_dict and s != sfGFP_WT:
        candidate_dict[s] = m
        if len(candidate_dict) % 500 == 0:
            print(f"  已生成 {len(candidate_dict)} 个...")
print(f"共生成 {len(candidate_dict)} 个候选。")

# ================== 5. 预测筛选 ==================
print("\n[5/5] 预测并筛选 Top 6...")
esm_pred, alpha_pred = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
esm_pred.eval().to(DEVICE)
conv_pred = alpha_pred.get_batch_converter()

cand_seqs = list(candidate_dict.keys())
cand_muts = [candidate_dict[s] for s in cand_seqs]
X_cand = get_embeddings(cand_seqs, esm_pred, conv_pred, DEVICE)
if X_cand.shape[0] == 0:
    raise ValueError("候选嵌入失败。")
pred_vals = rf.predict(X_cand)

# --- 修复：自动检测 Exclusion_List 列名 ---
exclusion_df = pd.read_csv(EXCLUSION_FILE)
possible_cols = ['Sequence', 'sequences-not-submit', 'sequence', 'seq']
col = None
for c in possible_cols:
    if c in exclusion_df.columns:
        col = c
        break
if col is None:
    raise ValueError(f"Exclusion_List.csv 中未找到序列列，现有列: {exclusion_df.columns.tolist()}")
exclusion_set = set(exclusion_df[col].astype(str))

results = pd.DataFrame({
    'Sequence': cand_seqs,
    'Mutations': cand_muts,
    'Pred_Brightness': pred_vals
})
results = results[~results['Sequence'].isin(exclusion_set)]
results = results[results['Sequence'].str.count('P') <= 15]
results = results.sort_values('Pred_Brightness', ascending=False)

final = results.head(TOP_N_SELECT).copy()
if final.empty:
    raise ValueError("无候选通过筛选。")

final['Team_Name'] = "YourTeamName"
final['Seq_ID'] = [str(i+1) for i in range(len(final))]

submission = final[['Team_Name', 'Seq_ID', 'Sequence']]
submission.to_csv('submission_2026.csv', index=False)
print("\n✅ 提交文件已生成: submission_2026.csv")
print("Top 6 序列:")
print(submission[['Seq_ID', 'Sequence']].head(6))
print("\n突变详情:")
print(final[['Seq_ID', 'Mutations', 'Pred_Brightness']])