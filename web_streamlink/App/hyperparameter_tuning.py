# App/hyperparameter_tuning.py
import pandas as pd
from itertools import product

# Константы из main.py
from ids_utils import BINNING_SPEC, SELECTED_COLS, FEATURE_GROUPS, discretize_columns, row_features

def run_grid_search(dataset_path: str,
                    template_path: str,
                    rules_path: str,
                    min_len_list: list[int],
                    min_rules_list: list[int]) -> pd.DataFrame:
    """
    Выполняет grid-search по параметрам min_len и min_rules,
    возвращает DataFrame с колонками ['min_len','min_rules','precision','recall','f1'].
    """
    # 1) Загрузка и дискретизация размеченного датасета
    df_l = pd.read_csv(dataset_path, parse_dates=["Timestamp"])
    df_disc = discretize_columns(df_l, BINNING_SPEC, SELECTED_COLS)

    # 2) Шаблон DDoS
    tmpl_df = pd.read_csv(template_path)
    ddos_template = set()
    for _, row in tmpl_df[tmpl_df["Attack"]=="DDOS"].iterrows():
        col, bins = row["Values"].split("=",1)
        for b in bins.split("|"):
            ddos_template.add(f"{col}={b}")

    print("Размер ddos_template =", len(ddos_template))
    print("Пример ddos_template — первые 5:", list(ddos_template)[:5])

    # 3) Ассоц. правила
    rf = pd.read_csv(rules_path)
    raw_rules = []
    for _, r in rf.iterrows():
        A = set(x.strip() for x in r["Antecedent"].strip("{}").split(","))
        B = r["Consequent"]
        raw_rules.append((A, B))
    print("Всего raw_rules =", len(raw_rules))
    print("Пример raw_rules[0] =", raw_rules[0])


    # 4) Grid search
    results = []
    for ml, mr in product(min_len_list, min_rules_list):
        # фильтрация правил
        filtered = [(A,B) for A,B in raw_rules
                    if len(A)>=ml and A.issubset(ddos_template) and (B in ddos_template)]
        # оценка
        TP=FP=TN=FN=0
        for idx, raw in df_l.iterrows():
            feats = row_features(raw, df_disc.loc[idx])
            ok = all(
                {f for f in ddos_template if any(f.startswith(c+"=") for c in cols)} & feats
                for grp, cols in FEATURE_GROUPS.items()
            )
            if not ok:
                pred = "BENIGN"
            else:
                fires = sum(1 for A,B in filtered if A.issubset(feats) and B in feats)
                pred = "DDoS" if fires >= mr else "BENIGN"
            true = raw["Label"]
            if pred == true:
                if pred=="DDoS": TP+=1
                else:             TN+=1
            else:
                if pred=="DDoS": FP+=1
                else:             FN+=1
        precision = TP/(TP+FP) if TP+FP else 0
        recall    = TP/(TP+FN) if TP+FN else 0
        f1        = 2*precision*recall/(precision+recall) if precision+recall else 0
        print(f"min_len={ml}, min_rules={mr} → filtered_rules={len(filtered)}")
        results.append({"min_len":ml, "min_rules":mr,
                        "precision":precision,
                        "recall":recall,
                        "f1":f1})

    return pd.DataFrame(results).sort_values(["f1","precision"], ascending=False).reset_index(drop=True)
