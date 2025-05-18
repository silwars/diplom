# App/test_ids.py
# Encapsulates full Test IDS logic including metrics and plotting

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from ids_utils import BINNING_SPEC, SELECTED_COLS, FEATURE_GROUPS, discretize_columns, row_features


def run_test_ids(dataset_path: str,
                 template_path: str,
                 rules_path: str,
                 min_rules_threshold: int,
                 min_len_thresh: int) -> dict:
    """
    Executes the IDS test and returns a dict with:
      - metrics: {precision, recall, TP, FP, TN, FN}
      - predictions: DataFrame of row-level predictions (including 'fires')
      - confusion_matrix: DataFrame or numpy array
      - fig_roc, fig_pr, fig_cm: matplotlib Figure objects
    """
    # 1) Load and discretize labeled dataset
    df_labeled = pd.read_csv(dataset_path, parse_dates=['Timestamp'])
    df_disc = discretize_columns(df_labeled, BINNING_SPEC, SELECTED_COLS)

    # 2) Load attack template
    tmpl_df = pd.read_csv(template_path)
    ddos_template = set()
    mask = tmpl_df['Attack'].str.upper() == 'DDOS'
    for _, row in tmpl_df[mask].iterrows():
        col, bins = row['Values'].split('=', 1)
        for b in bins.split('|'):
            ddos_template.add(f"{col}={b}")

    print(f"В файле шаблона найдено строк для DDoS: {ddos_template}")

    # 3) Load rules
    rules_df = pd.read_csv(rules_path)
    raw_rules = []
    for _, r in rules_df.iterrows():
        A = set(x.strip() for x in r['Antecedent'].strip('{}').split(','))
        B = r['Consequent']
        raw_rules.append((A, B))

    # 4) Filter rules by template and min antecedent length
    filtered_rules = [
        (A, B)
        for (A, B) in raw_rules
        if len(A) >= min_len_thresh
           and A.issubset(ddos_template)
           and (B in ddos_template)
    ]
    print(f"После фильтрации по шаблону и длине A ≥ {min_len_thresh}: найдено {len(filtered_rules)} правил")


    # 5) Detection loop
    preds = []
    TP = FP = TN = FN = 0
    for idx, raw in df_labeled.iterrows():
        feats = row_features(raw, df_disc.loc[idx])
        ok = all(
            {f for f in ddos_template if any(f.startswith(c+'=') for c in cols)} & feats
            for cols in FEATURE_GROUPS.values()
        )
        if not ok:
            fires = 0
            pred = 'BENIGN'
        else:
            fires = sum(1 for A, B in filtered_rules if A.issubset(feats) and B in feats)
            pred = 'DDoS' if fires >= min_rules_threshold else 'BENIGN'

        true = raw['Label']
        if pred == true:
            if pred == 'DDoS':
                TP += 1
            else:
                TN += 1
        else:
            if pred == 'DDoS':
                FP += 1
            else:
                FN += 1

        rec = raw.to_dict()
        rec.update({'Predicted': pred, 'fires': fires})
        preds.append(rec)

    # Metrics
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    # Predictions DataFrame
    df_pred = pd.DataFrame(preds)

    # 6) Prepare ROC figure
    y_true = (df_pred['Label'] == 'DDoS').astype(int).values
    y_score = df_pred['fires'].values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 3))
    ax_roc.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    ax_roc.plot([0, 1], [0, 1], linestyle='--')
    ax_roc.set_title('ROC кривая')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right')

    # 7) Precision–Recall figure
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_score)
    fig_pr, ax_pr = plt.subplots(figsize=(5, 3))
    ax_pr.plot(recall_arr, precision_arr, marker='.')
    ax_pr.set_title('Precision–Recall кривая')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')

    # 8) Confusion matrix figure
    y_pred = (df_pred['Predicted'] == 'DDoS').astype(int).values
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
    cax = ax_cm.matshow(cm, cmap='Blues')
    fig_cm.colorbar(cax)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['BENIGN', 'DDoS'])
    ax_cm.set_yticklabels(['BENIGN', 'DDoS'])
    ax_cm.xaxis.set_ticks_position('bottom')
    ax_cm.xaxis.tick_bottom()
    ax_cm.set_xlabel('Предсказание')
    ax_cm.set_ylabel('Истина')
    ax_cm.set_title('Матрица ошибок')

    return {
        'metrics': {
            'precision': precision,
            'recall':    recall,
            'TP':        TP,
            'FP':        FP,
            'TN':        TN,
            'FN':        FN
        },
        'predictions':      df_pred,
        'confusion_matrix': cm,
        'fig_roc':          fig_roc,
        'fig_pr':           fig_pr,
        'fig_cm':           fig_cm
    }
