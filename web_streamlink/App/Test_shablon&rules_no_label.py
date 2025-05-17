#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1. Импорты и константы
import pandas as pd
import numpy as np
from datetime import timedelta

# параметры дискретизации
BINNING_SPEC = {
    'Total Fwd Packets':      ('uniform', 5),
    'Total Backward Packets': ('uniform', 5),
    'Total Length of Fwd Packets': ('uniform', 5),
    'Total Length of Bwd Packets': ('uniform', 5),
    'Active Mean':            ('uniform', 5),
    'Idle Mean':              ('uniform', 5),
    'Flow Duration':          ('uniform', 5)
}

SELECTED_COLS = [
    'Total Fwd Packets','Total Backward Packets',
    'Total Length of Fwd Packets','Total Length of Bwd Packets',
    'Active Mean','Idle Mean','Flow Duration',
    'Destination IP','Destination Port',
    'FIN Flag Count','SYN Flag Count',
    'RST Flag Count','PSH Flag Count','ACK Flag Count'
]

FEATURE_GROUPS = {
    'network': [
        'Destination IP','Destination Port',
        'FIN Flag Count','SYN Flag Count',
        'RST Flag Count','PSH Flag Count','ACK Flag Count'
    ],
    'temporal': ['Flow Duration'],
    'traffic': [
        'Total Fwd Packets','Total Backward Packets',
        'Total Length of Fwd Packets','Total Length of Bwd Packets',
        'Active Mean','Idle Mean'
    ]
}

# Порог срабатывания: сколько отфильтрованных правил нужно «пробить»
MIN_RULES = 150


# In[2]:


# Cell 2. Функции дискретизации и извлечения фич для одной строки

def discretize_columns(df_in, spec, cols):
    df2 = df_in[['Timestamp'] + cols].copy()
    for col, (method, bins) in spec.items():
        if col not in cols: 
            continue
        arr = df2[col].astype(float)
        labels = [f"bin{i}" for i in range(bins)]
        if method == 'quantile':
            df2[col] = pd.qcut(arr, q=bins, labels=labels, duplicates='drop').astype(str)
        else:
            df2[col] = pd.cut(arr, bins=bins, labels=labels).astype(str)
    return df2

def row_features(raw, disc):
    """
    Собирает все SELECTED_COLS в формат 'Feature=val' для одной строки.
    """
    feats = set()
    for col in SELECTED_COLS:
        if col in BINNING_SPEC:
            feats.add(f"{col}={disc[col]}")
        else:
            feats.add(f"{col}={raw[col]}")
    return feats


# In[3]:


# Cell 3. Загрузка и дискретизация датасетов

# 3.1 размеченный (для оценки)
df_labeled = pd.read_csv(
    r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_sampled1.csv',
    parse_dates=['Timestamp']
)

# 3.2 неразмеченный (для реального теста, но здесь не нужен)
df_unlabeled = pd.read_csv(
    r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_sampled2.csv',
    parse_dates=['Timestamp']
)

# 3.3 дискретизируем оба
df_disc_labeled   = discretize_columns(df_labeled,   BINNING_SPEC, SELECTED_COLS)
df_disc_unlabeled = discretize_columns(df_unlabeled, BINNING_SPEC, SELECTED_COLS)


# In[4]:


# Cell 4. Загрузка Top-K-шаблона и сбор DDoS-фич

tmpl_df = pd.read_csv(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\generated_attack_templates.csv')  # поля: Attack, Group, Values
ddos_template = set()

for _, row in tmpl_df[tmpl_df['Attack']=='DDOS'].iterrows():
    # 'Values' может быть "Flow Duration=bin0|bin3" или "Destination IP=192.168.10.50"
    col, bins = row['Values'].split('=',1)
    for b in bins.split('|'):
        ddos_template.add(f"{col}={b}")


# In[5]:


# ─── Cell 5. Загрузка неразмеченных правил и фильтрация по шаблону ─────────────



# читаем AR-правила без меток
rules_df = pd.read_csv(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\association_rules_no_label.csv')  # Antecedent, Consequent, Support, Confidence, Lift
raw_rules = []
for _, r in rules_df.iterrows():
    A = set(x.strip() for x in r['Antecedent'].strip('{}').split(','))
    B = r['Consequent']
    sup  = r['Support']
    conf = r['Confidence']
    lift = r['Lift']
    raw_rules.append((A, B, sup, conf, lift))
    
    # теперь фильтруем по шаблону и по длине A ≥ 2
filtered_rules = [
    (A, B)
    for (A, B, sup, conf, lift) in raw_rules
    # правило должно состоять минимум из 2 признаков
    if len(A) >= 4
    # все его фичи лежат в DDoS-шаблоне
    and A.issubset(ddos_template)
    # и сам consequent должен быть в шаблоне
    and (B in ddos_template)
]

print(f"Всего ассоциативных правил: {len(raw_rules)}, после фильтрации по шаблону: {len(filtered_rules)}")


# In[6]:


# Cell 6. Детекция по строкам и оценка качества

preds = []
TP = FP = TN = FN = 0

for idx, raw in df_labeled.iterrows():
    disc  = df_disc_labeled.loc[idx]
    feats = row_features(raw, disc)

    # обязательная проверка: хотя бы по одному фичу из каждой группы
    ok = True
    for grp, cols in FEATURE_GROUPS.items():
        # берем из шаблона только фичи этой группы
        grp_feats = {f for f in ddos_template if any(f.startswith(c+"=") for c in cols)}
        if not (grp_feats & feats):
            ok = False
            break

    if not ok:
        fires = 0
        pred = 'BENIGN'
    else:
        # считаем, сколько отфильтрованных правил ≡ A⇒B 'прошли'
        fires = sum(1 for A,B in filtered_rules if A.issubset(feats) and (B in feats))
        pred  = 'DDoS' if fires >= MIN_RULES else 'BENIGN'

    true = raw['Label']
    if pred == true:
        if pred == 'DDoS': TP += 1
        else:             TN += 1
    else:
        if pred == 'DDoS': FP += 1
        else:             FN += 1

    rec = raw.to_dict()
    rec['Predicted'] = pred
    rec['fires']     = fires
    preds.append(rec)

precision = TP / (TP + FP) if TP + FP else 0
recall    = TP / (TP + FN) if TP + FN else 0
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")

# сохраняем все строки с предсказаниями
pd.DataFrame(preds).to_csv(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\row_level_predictions.csv', index=False)
#print("Готово: row_level_predictions.csv")

