#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell X: Build and Save Extended Top-K Attack Templates

import pandas as pd
import csv

# ——————————————————————————————
# Константы и параметры
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

TOP_K           = 3      # сколько самых частых бинов брать
ABS_SUPPORT     = 400    # минимум 400 вхождений в DDoS
BENIGN_MAX_FREQ = 0.90   # допускаем, что бин встречается до 90% в Benign
FORCE_HIGH_FREQ = 0.80   # если бин ≥80% в DDoS — держим его вне зависимости от Benign

# ——————————————————————————————
# Функция дискретизации (копия из вашего скрипта)
def discretize_columns(df_in, spec, cols):
    df2 = df_in[['Timestamp'] + cols].copy()
    for c, (method, bins) in spec.items():
        if c not in cols:
            continue
        arr = df2[c].astype(float)
        labels = [f"bin{i}" for i in range(bins)]
        if method == 'quantile':
            b = pd.qcut(arr, q=bins, labels=labels, duplicates='drop')
        else:
            b = pd.cut(arr, bins=bins, labels=labels)
        df2[c] = b.astype(str)
    return df2

# ——————————————————————————————
# 1) Загружаем размеченный датасет
df = pd.read_csv(
    r'C:\Users\Гребенников Матвей\Desktop\Диплом\Диплом\Code\diplom-project\diplom\test\result\Date\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_sampled_v3.csv',
    parse_dates=['Timestamp']
)

# 2) Дискретизируем все нужные столбцы
df_disc = discretize_columns(df, BINNING_SPEC, SELECTED_COLS)

# 3) Список атак (все, кроме BENIGN)
attacks = [lbl for lbl in df['Label'].unique() if lbl != 'BENIGN']

attack_templates = []

# 4) Для каждой атаки и для каждой группы и для каждого столбца
for att in attacks:
    df_att = df_disc[df['Label'] == att]
    df_ben = df_disc[df['Label'] == 'BENIGN']

    for grp, cols in FEATURE_GROUPS.items():
        for col in cols:
            # a) частоты в DDoS
            cnt_att  = df_att[col].value_counts()
            top_bins = list(cnt_att.index[:TOP_K])

            # b) частоты в Benign (нормированные)
            freq_ben = df_ben[col].value_counts(normalize=True)

            # c) отбираем только те бины, которые ≥ ABS_SUPPORT в DDoS
            good = [b for b in top_bins if cnt_att.get(b, 0) >= ABS_SUPPORT]

            # d) фильтрация по Benign, но «спасаем» высокочастотные
            filtered = []
            for b in good:
                p_ben = freq_ben.get(b, 0)
                p_att = cnt_att[b] / len(df_att)
                if p_ben <= BENIGN_MAX_FREQ or p_att >= FORCE_HIGH_FREQ:
                    filtered.append(b)
            good = filtered

            # e) fallback — если в DDoS есть хоть один хороший, но его отфильтровали,
            #    вернём самый частый бин
            if not good and cnt_att.get(top_bins[0], 0) >= ABS_SUPPORT:
                good = [top_bins[0]]

            # f) наконец, если после всего осталось хотя бы что-то — сохраняем
            if good:
                attack_templates.append({
                    'Attack': att,
                    'Group': grp,
                    'Values': f"{col}=" + "|".join(str(x) for x in good)
                })

# 5) Сохраняем в CSV
with open('attack_templates_topk_extended_test.csv','w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['Attack','Group','Values'])
    w.writeheader()
    w.writerows(attack_templates)

print("attack_templates_topk_extended.csv ready.")

