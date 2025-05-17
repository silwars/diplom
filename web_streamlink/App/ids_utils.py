# App/ids_utils.py

import pandas as pd

# ─── Константы ────────────────────────────────────────
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

# ─── Функции ──────────────────────────────────────────
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
