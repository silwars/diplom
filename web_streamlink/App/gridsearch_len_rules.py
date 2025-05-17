# gridsearch_len_rules.py
# Standalone grid search for IDS associative rules parameters

import pandas as pd
from itertools import product
from ids_utils import BINNING_SPEC, SELECTED_COLS, FEATURE_GROUPS, discretize_columns, row_features

# ─── Константы и параметры ────────────────────────────────────────────────
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

# Параметры grid search
min_len_list   = [2, 3, 4, 5]
min_rules_list = [50, 100, 150, 200, 250]

# ─── Функции вспомогательные ────────────────────────────────────────────────

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
    feats = set()
    for col in SELECTED_COLS:
        if col in BINNING_SPEC:
            feats.add(f"{col}={disc[col]}")
        else:
            feats.add(f"{col}={raw[col]}")
    return feats

# ─── Загрузка и подготовка данных ──────────────────────────────────────────
print("Loading labeled dataset and discretizing...")
# Укажите здесь путь к размеченному CSV
df_labeled = pd.read_csv(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_sampled1.csv', parse_dates=['Timestamp'])
df_disc_labeled = discretize_columns(df_labeled, BINNING_SPEC, SELECTED_COLS)

# Загрузка шаблона DDoS
tmpl_df = pd.read_csv(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\generated_attack_templates.csv')
ddos_template = set()
for _, row in tmpl_df[tmpl_df['Attack']=='DDOS'].iterrows():
    col, bins = row['Values'].split('=',1)
    for b in bins.split('|'):
        ddos_template.add(f"{col}={b}")

# Загрузка сырого списка правил
rules_df = pd.read_csv(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\association_rules_no_label.csv')
raw_rules = []
for _, r in rules_df.iterrows():
    A = set(x.strip() for x in r['Antecedent'].strip('{}').split(','))
    B = r['Consequent']
    sup, conf, lift = r['Support'], r['Confidence'], r['Lift']
    raw_rules.append((A, B, sup, conf, lift))

# ─── Grid Search ──────────────────────────────────────────────────────────
results = []
print("Starting grid search...")
for min_len, min_rules in product(min_len_list, min_rules_list):
    # Фильтрация правил по длине и шаблону
    filtered = [
        (A, B)
        for (A, B, sup, conf, lift) in raw_rules
        if len(A) >= min_len and A.issubset(ddos_template) and (B in ddos_template)
    ]

    # Оценка качества по предсказаниям
    TP = FP = TN = FN = 0
    for idx, raw in df_labeled.iterrows():
        disc = df_disc_labeled.loc[idx]
        feats = row_features(raw, disc)
        # Проверка групп
        ok = all(
            set(f for f in ddos_template if any(f.startswith(c+"=") for c in cols)) & feats
            for grp, cols in FEATURE_GROUPS.items()
        )
        if not ok:
            pred = 'BENIGN'
        else:
            fires = sum(1 for A,B in filtered if A.issubset(feats) and (B in feats))
            pred = 'DDoS' if fires >= min_rules else 'BENIGN'

        true = raw['Label']
        if pred == true:
            if pred == 'DDoS': TP += 1
            else:             TN += 1
        else:
            if pred == 'DDoS': FP += 1
            else:             FN += 1

    precision = TP/(TP+FP) if TP+FP else 0
    recall    = TP/(TP+FN) if TP+FN else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    results.append({
        'min_len': min_len,
        'min_rules': min_rules,
        'num_rules': len(filtered),
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'precision': precision, 'recall': recall, 'f1': f1
    })

# Сборка итогового DataFrame
df_res = pd.DataFrame(results).sort_values(['f1','precision'], ascending=False).reset_index(drop=True)
print(df_res)

# Сохранение
out_path = r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\gridsearch_len_rules.csv'
df_res.to_csv(out_path, index=False)
print(f"Grid search results saved to {out_path}")
