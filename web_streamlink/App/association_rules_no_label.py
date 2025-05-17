#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1: Импорт, параметры и выбираемые заголовки
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from datetime import timedelta
import csv
from multiprocessing import Pool, cpu_count
import time

print("[Init] Libraries imported.")

# Window parameters
window_size = timedelta(minutes=30)
slide_size  = timedelta(minutes=15)
print(f"[Init] Window size: {window_size}, Slide size: {slide_size}")

# Discretization specification
BINNING_SPEC = {
    'Total Fwd Packets': ('uniform', 5),
    'Total Backward Packets': ('uniform', 5),
    'Total Length of Fwd Packets': ('uniform', 5),
    'Total Length of Bwd Packets': ('uniform', 5),
    'Active Mean': ('uniform', 5),
    'Idle Mean': ('uniform', 5),
    'Flow Duration': ('uniform', 5)
}

# Параметры алгоритма
MIN_SUPPORT = 10
MIN_CONF    = 0.9
ALPHA       = 3
print(f"[Init] MIN_SUPPORT={MIN_SUPPORT}, MIN_CONF={MIN_CONF}, ALPHA={ALPHA}")

# Группы для MDFP
FEATURE_GROUPS = {
    'network': ['Destination IP','Destination Port','FIN Flag Count','SYN Flag Count',
                'RST Flag Count','PSH Flag Count','ACK Flag Count'],
    'temporal': ['Timestamp','Flow Duration'],
    'traffic': ['Total Fwd Packets','Total Backward Packets',
                'Total Length of Fwd Packets','Total Length of Bwd Packets',
                'Active Mean','Idle Mean']
}
PRIMARY_FEATURES = set(f for grp in FEATURE_GROUPS.values() for f in grp)
print(f"[Init] PRIMARY_FEATURES count: {len(PRIMARY_FEATURES)}")


SELECTED_COLS = ['Total Fwd Packets','Total Backward Packets',
                'Total Length of Fwd Packets','Total Length of Bwd Packets',
                'Active Mean','Idle Mean', 'Flow Duration', 'Destination IP','Destination Port','FIN Flag Count','SYN Flag Count',
                'RST Flag Count','PSH Flag Count','ACK Flag Count'
]

# Флаги для отключения спама принами
VERBOSE_MDFP   = False
MDFP_PRINTED   = False


# In[2]:


# Cell 2: Загрузка и обработка датасета

path = r'C:\Users\Гребенников Матвей\Desktop\Диплом\Диплом\Code\diplom-project\diplom\test\result\Date\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_sampled_v4.csv'
df = pd.read_csv(path, low_memory=False)
print(f"[Load] Read DataFrame from {path}, shape: {df.shape}")

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
print("[Load] Converted 'Timestamp' to datetime.")
df.sort_values('Timestamp', inplace=True)
print("[Load] Sorted DataFrame by Timestamp.")
df.reset_index(drop=True, inplace=True)
print("[Load] Reset DataFrame index.")


# In[3]:


# Cell 3: Generate and Encode Transactions (с метками bin0…binN вместо интервалов)

def discretize_columns(df_in, spec, cols):
    """
    Оставляем только Timestamp и указанные cols, 
    дискретизируем с uniform/quantile и метками bin0, bin1, … bin{N-1}
    """
    df2 = df_in[['Timestamp'] + cols].copy()
    for col, (method, bins) in spec.items():
        if col not in df2 or col not in cols:
            continue
        arr = df2[col].astype(float)
        # формируем метки bin0…bin{bins-1}
        labels = [f"bin{i}" for i in range(bins)]
        if method == 'quantile':
            binned = pd.qcut(arr, q=bins, labels=labels, duplicates='drop')
        else:  # uniform
            binned = pd.cut(arr, bins=bins, labels=labels)
        df2[col] = binned.astype(str).astype('category')
    return df2

# Build string transactions
str_trans = []
df_disc = discretize_columns(df, BINNING_SPEC, SELECTED_COLS)
print(f"[Trans] Discretized selected columns, shape: {df_disc.shape}")
for _, row in df_disc.iterrows():
    tr = []
    for col in SELECTED_COLS:
        val = row[col]
        # фильтруем нулевые флаги, если нужно
        if 'flag count' in col.lower() and str(val).startswith('0'):
            continue
        tr.append(f"{col}:{val}")   # e.g. "Active Mean:bin2"
    str_trans.append(tr)
print(f"[Trans] Built string transactions, count: {len(str_trans)}")

# Encode to integer IDs
all_items = sorted({it for tr in str_trans for it in tr})
item2id = {it:i for i,it in enumerate(all_items)}
id2item = {i:it for it,i in item2id.items()}
print(f"[Trans] Unique items: {len(all_items)}")
# Global integer transactions
global_trans = [[item2id[it] for it in tr] for tr in str_trans]
print(f"[Trans] Encoded global transactions, count: {len(global_trans)}")


# Function for windowed transactions (аналогично, с метками binX)
def make_transactions_ids(df_window):
    dfw = discretize_columns(df_window, BINNING_SPEC, SELECTED_COLS)
    trans = []
    for _, row in dfw.iterrows():
        tr = []
        for col in SELECTED_COLS:
            if 'flag count' in col.lower() and str(row[col]).startswith('0'):
                continue
            item = f"{col}:{row[col]}"   # e.g. "Active Mean:bin4"
            if item in item2id:
                tr.append(item2id[item])
        trans.append(tr)
    return trans

print("[Trans] Prepared make_transactions_ids function.")


# In[4]:


# Cell 4: COFI-Tree 
class COFINode:
    __slots__ = ('item','count','parent','children','node_link')
    def __init__(self, item, parent):
        self.item = item
        self.count = 1
        self.parent = parent
        self.children = {}
        self.node_link = None
def create_cofi_tree(transactions, min_sup):
    freq = defaultdict(int)
    for tr in transactions:
        for it in tr:
            freq[it] += 1
    freq = {it:c for it,c in freq.items() if c>=min_sup}
    if not freq:
        return None, None
    header = {it:[freq[it], None, None] for it in freq}
    root = COFINode(None, None)
    for tr in transactions:
        flt = [it for it in tr if it in freq]
        flt.sort(key=lambda x: freq[x], reverse=True)
        node = root
        for it in flt:
            if it in node.children:
                child = node.children[it]
                child.count += 1
            else:
                child = COFINode(it, node)
                node.children[it] = child
                if header[it][1] is None:
                    header[it][1] = child
                    header[it][2] = child
                else:
                    header[it][2].node_link = child
                    header[it][2] = child
            node = child
    return root, header


# In[5]:


# Cell 5: Оптимизированный COFI Mining 

from multiprocessing.dummy import Pool  
from multiprocessing import cpu_count
from collections import defaultdict

print("[COFI] Ready to mine prefixes (fully optimized).")

def find_cofi_prefixes(item, header):
    paths = {}
    node = header[item][1]
    while node:
        cnt = node.count
        prefix = []
        p = node.parent
        while p and p.item is not None:
            prefix.append(p.item)
            p = p.parent
        if prefix:
            paths[frozenset(prefix)] = paths.get(frozenset(prefix), 0) + cnt
        node = node.node_link
    return paths

def iterate_node_links(item, header):
    node = header[item][1]
    while node:
        yield node
        node = node.node_link

def build_conditional_cofi_tree(base, header, min_sup):
    
    freq = defaultdict(int)
    conditional_paths = []
    for node in iterate_node_links(base, header):
        path = []
        p = node.parent
        while p and p.item is not None:
            path.append(p.item)
            p = p.parent
        if path:
            conditional_paths.append((path, node.count))
            for it in path:
                freq[it] += node.count

    # 2) Filter infrequent items
    freq = {it:c for it,c in freq.items() if c >= min_sup}
    if not freq:
        return None, None

    # 3) Initialize new header and root
    cond_header = {it:[freq[it], None, None] for it in freq}
    root = COFINode(None, None)

    # 4) Insert each conditional path into the new COFI-tree
    for path, cnt in conditional_paths:
        flt = [it for it in path if it in freq]
        if not flt: continue
        flt.sort(key=lambda x: freq[x], reverse=True)
        node = root
        for it in flt:
            if it in node.children:
                child = node.children[it]
                child.count += cnt
            else:
                child = COFINode(it, node)
                child.count = cnt
                node.children[it] = child
                # back-pointer insertion
                if cond_header[it][1] is None:
                    cond_header[it][1] = child
                    cond_header[it][2] = child
                else:
                    cond_header[it][2].node_link = child
                    cond_header[it][2] = child
            node = child

    return root, cond_header

def rec_cofi(root, header, min_sup, prefix, patterns, max_depth, base, order):
    if len(prefix) >= max_depth:
        return
    for it, (sup, _, _) in sorted(header.items(), key=lambda x: x[1][0]):
        if it <= base or order[it] <= order[base]:
            continue
        newp = prefix | {it}
        patterns[frozenset(newp)] = sup
        # build and mine conditional tree recursively
        cond_root, cond_hdr = build_conditional_cofi_tree(it, header, min_sup)
        if cond_hdr:
            rec_cofi(cond_root, cond_hdr, min_sup, newp, patterns, max_depth, it, order)

def mine_cofi_base(args):
    base, root, header, min_sup, max_depth, order = args
    print(f"[COFI-base] Start base {base}")
    patterns = {frozenset([base]): header[base][0]}

    # Пропуск по минимальной поддержке
    total_support = sum(node.count for node in iterate_node_links(base, header))
    if total_support < min_sup:
        print(f"[COFI-base] Base {base}: conditional support {total_support} < {min_sup}, skipping")
        print(f"[COFI-base] Finished base {base}, found {len(patterns)} patterns")
        return patterns

    cond_root, cond_hdr = build_conditional_cofi_tree(base, header, min_sup)
    if cond_hdr:
        rec_cofi(cond_root, cond_hdr, min_sup, {base}, patterns, max_depth, base, order)

    print(f"[COFI-base] Finished base {base}, found {len(patterns)} patterns")
    return patterns

# Глобальный COFI
root_c, hdr_c = create_cofi_tree(global_trans, MIN_SUPPORT)
if hdr_c:
    print(f"[COFI] Built global tree: {len(hdr_c)} frequent items")
    bases = sorted(hdr_c)
    order = {b:i for i,b in enumerate(bases)}
    args = [(b, root_c, hdr_c, MIN_SUPPORT, 4, order) for b in bases]
    print(f"[COFI] Mining {len(bases)} base elements via ThreadPool...")
    patterns_global = {}
    with Pool(min(cpu_count(), len(bases))) as pool:
        for part in pool.map(mine_cofi_base, args):
            patterns_global.update(part)
    valid_global = set().union(*patterns_global.keys())
    print(f"[COFI] Global mining done: {len(patterns_global)} patterns, {len(valid_global)} valid elements")
else:
    print(f"[COFI] No frequent items >= MIN_SUPPORT={MIN_SUPPORT}")


# In[6]:


# Cell 6: MDFP-Tree Construction с параллельным подсчётом пар

from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool, cpu_count

# (флаги и класс MDFPNode остаются без изменений)

def _count_pairs_chunk(args):
    chunk, freq = args
    local = defaultdict(lambda: defaultdict(int))
    for tr in chunk:
        flt = [it for it in tr if it in freq]
        # НЕ сортируем — combinations пройдёт по всем парам в любом порядке
        for i,j in combinations(flt,2):
            local[i][j] += 1
            local[j][i] += 1
    return local
class MDFPNode:
    __slots__ = ('item', 'group', 'parent', 'children', 'count', 'link', 'array')
    def __init__(self, item, group, parent):
        self.item     = item
        self.group    = group
        self.parent   = parent
        self.children = {}
        self.count    = 1
        self.link     = None
        self.array    = 0

def create_mdfp_tree(transactions, min_sup):
    """
    1) Считаем одиночные частоты
    2) Фильтруем по min_sup и готовим header
    3) Последовательно считаем pair_sup
    4) Строим дерево
    """
    # --- 1) одиночные частоты + header ---
    freq = defaultdict(int)
    for tr in transactions:
        for it in tr:
            freq[it] += 1
    freq = {it:c for it,c in freq.items() if c >= min_sup}
    if not freq:
        return None, None, {}
    header = {it:[freq[it], None, None, 0] for it in freq}

    # --- 2) последовательный подсчёт пар ---
    pair_sup = defaultdict(lambda: defaultdict(int))
    for tr in transactions:
        # фильтруем сразу по частотам
        flt = [it for it in tr if it in freq]
        # считаем для каждой пары
        for i,j in combinations(flt, 2):
            pair_sup[i][j] += 1
            pair_sup[j][i] += 1

    # --- 3) строим сам MDFP-дерево ---
    root = MDFPNode(None, None, None)
    for tr in transactions:
        flt = [it for it in tr if it in freq]
        if not flt:
            continue
        node = root
        for key in flt:
            if key in node.children:
                child = node.children[key]
                child.count += 1
            else:
                child = MDFPNode(
                    key,
                    next((g for g,v in FEATURE_GROUPS.items()
                          if id2item[key].split(':')[0] in v), None),
                    node
                )
                node.children[key] = child
                # back-pointer
                if header[key][1] is None:
                    header[key][1] = child
                    header[key][2] = child
                else:
                    header[key][2].link = child
                    header[key][2] = child
                header[key][3] += 1
            node = child

    return root, header, pair_sup


# Cell 6b: MDFP Mining Functions

def find_prefix_paths(base, header):
    """
    Из header[base] по ссылкам link собираем все префикс-пути и суммируем их count.
    """
    paths = {}
    node = header[base][1]
    while node:
        cnt = node.count
        prefix = []
        p = node.parent
        while p and p.item is not None:
            prefix.append(p.item)
            p = p.parent
        if prefix:
            paths[frozenset(prefix)] = paths.get(frozenset(prefix), 0) + cnt
        node = node.link
    return paths

def mine_sparse(item, header, pair_sup, transactions, min_sup, k_max=4):
    """
    Разреженная ветвь MDFP: генерируем L1, Lk по pair_sup без дерева.
    """
    patterns = {}
    L1 = [j for j in pair_sup[item] if pair_sup[item][j] >= min_sup]
    for j in L1:
        patterns[frozenset([item, j])] = pair_sup[item][j]
    Lk_1 = [frozenset([item, j]) for j in L1]
    k = 3
    while Lk_1 and k <= k_max:
        Ck = set()
        for prev in Lk_1:
            for j in L1:
                if j not in prev:
                    c = prev | {j}
                    if len(c) == k:
                        Ck.add(c)
        Lk = []
        for c in Ck:
            cnt = sum(1 for tr in transactions if c.issubset(tr))
            if cnt >= min_sup:
                patterns[c] = cnt
                Lk.append(c)
        Lk_1, k = Lk, k+1
    return patterns

def mine_dense_item(item, root, header, min_sup):
    """
    Плотная ветвь MDFP: строим условный набор транзакций и рекурсивно майним.
    """
    patterns = {frozenset([item]): header[item][0]}
    cond = find_prefix_paths(item, header)
    trans = []
    for pth, cnt in cond.items():
        trans += [list(pth)] * cnt
    if trans:
        sub_root, sub_header, _ = create_mdfp_tree(trans, min_sup)
        if sub_header:
            def rec(nr, nh, prefix):
                for it, (s, head, last, link_ct) in sorted(nh.items(), key=lambda x: x[1][0]):
                    newset = prefix | {it}
                    patterns[frozenset(newset)] = s
                    cond2 = find_prefix_paths(it, nh)
                    t2 = []
                    for p2, c2 in cond2.items():
                        t2 += [list(p2)] * c2
                    if t2:
                        sr, sh, _ = create_mdfp_tree(t2, min_sup)
                        if sh:
                            rec(sr, sh, newset)
            rec(sub_root, sub_header, {item})
    return patterns

def mine_mdfp(transactions, min_sup):
    """
    Основная точка входа для майнинга MDFP: строим дерево, 
    разделяем элементы на sparse/dense и собираем все patterns.
    """
    root, header, pair_sup = create_mdfp_tree(transactions, min_sup)
    if not header:
        return {}
    # разделяем на разреженные и плотные
    sparse = [it for it, (sup, head, last, link_ct) in header.items() if head and head.array == 1]
    dense  = [it for it in header if it not in sparse]
    patterns = {}
    for it in sparse:
        patterns.update(mine_sparse(it, header, pair_sup, transactions, min_sup))
    for it in dense:
        patterns.update(mine_dense_item(it, root, header, min_sup))
    return patterns


# In[7]:


# Cell 7: Pattern Filtering and Rule Generation

def filter_closed(patterns):
    closed = {}
    for p, s in patterns.items():
        if not any(len(q) > len(p) and p.issubset(q) and patterns[q] == s
                   for q in patterns):
            closed[p] = s
    print(f"[Filter] Closed patterns count: {len(closed)}")
    return closed

def generate_rules(patterns, min_conf, total_tx):
    rules = []
    for pat, supAB in patterns.items():
        # теперь требуем размер паттерна ≥ 3 (т.е. ≥2 в antecedent + 1 в consequent)
        if len(pat) < 3 or len(pat) > 5:
            continue

        # перебираем r-элементные A, где r от 2 до len(pat)-1
        for r in range(2, len(pat)):
            for A in combinations(pat, r):
                B = set(pat) - set(A)
                if len(B) != 1:
                    continue
                b = next(iter(B))

                # оставляем только A, которые содержат хотя бы один первичный признак
                if not any(id2item[a].split(':')[0] in PRIMARY_FEATURES for a in A):
                    continue

                supA = patterns.get(frozenset(A), 0)
                if supA == 0:
                    continue

                conf = supAB / supA
                if conf < min_conf:
                    continue

                supB = patterns.get(frozenset([b]), supAB)
                lift = conf / (supB / total_tx) if supB > 0 else np.nan

                # фильтруем lift, близкий к 1: |lift − 1| ≤ 0.15
                if abs(lift - 1) <= 0.15:
                    continue

                rules.append((set(A), b, supAB, conf, lift))

    # удаляем избыточные правила
    final = []
    for i, (A, b, s, cf, l) in enumerate(rules):
        if not any(
            i != j and b == b2 and s == s2 and abs(cf - cf2) < 1e-9 and A2.issubset(A)
            for j, (A2, b2, s2, cf2, l2) in enumerate(rules)
        ):
            final.append((A, b, s, cf, l))

    print(f"[Rules] Generated {len(final)} filtered rules")
    return final


# In[8]:


# Cell 8: Sliding Window Loop and CSV Export

import time
import csv

# helper: превращаем "Col:val" → "Col=val" (без _bin)
def format_item(item_str):
    col, val = item_str.split(':', 1)
    return f"{col}={val}"

# Инициализация CSV-файлов
with open(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\frequent_itemsets_no_label.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['Start','End','Pattern','Support'])
with open(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\closed_patterns_no_label.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['Start','End','ClosedPattern','Support'])
with open(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\association_rules_no_label.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['Start','End','Antecedent','Consequent','Support','Confidence','Lift'])
print("[CSV] Initialized CSV files.")

# Генерируем окна
windows = []
cur = df['Timestamp'].min()
end = df['Timestamp'].max()
while cur + window_size <= end:
    windows.append((cur, cur + window_size))
    cur += slide_size
print(f"[Window] Total windows: {len(windows)}")

# Обработка по окнам
for idx, (t0, t1) in enumerate(windows, 1):
    print(f"\n[Window {idx}/{len(windows)}] {t0} -> {t1}")
    df_w = df[(df['Timestamp'] >= t0) & (df['Timestamp'] < t1)]
    trans = make_transactions_ids(df_w)
    print(f"[Window {idx}] Transactions: {len(trans)}")
    filt = [[i for i in tr if i in valid_global] for tr in trans]
    non_empty = sum(1 for tr in filt if tr)
    print(f"[Window {idx}] After filter: {non_empty} non-empty")

    # майнинг MDFP с таймером
    print(f"[Window {idx}] Starting MDFP mining…")
    t_start = time.time()
    mdfp_pats = mine_mdfp(filt, MIN_SUPPORT)
    t_end = time.time()
    print(f"[Window {idx}] MDFP done in {t_end-t_start:.1f}s, patterns: {len(mdfp_pats)}")

    closed = filter_closed(mdfp_pats)
    rules  = generate_rules(closed, MIN_CONF, non_empty)
    print(f"[Window {idx}] Rules: {len(rules)}")

    # Export frequent patterns
    with open(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\frequent_itemsets_no_label.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for p, sup in mdfp_pats.items():
            pattern_str = '[' + ', '.join(format_item(id2item[i]) for i in p) + ']'
            writer.writerow([t0, t1, pattern_str, sup])

    # Export closed patterns (sd) with support
    with open(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\closed_patterns_no_label.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for p, sup in closed.items():
            pattern_str = '{' + ', '.join(format_item(id2item[i]) for i in p) + '}'
            writer.writerow([t0, t1, pattern_str, sup])

    # Export association rules with full columns
    with open(r'C:\Users\Гребенников Матвей\Desktop\WEB_TEST\data\association_rules_no_label.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for A, b, supAB, conf, lift in rules:
            ants = '{' + ', '.join(format_item(id2item[i]) for i in A) + '}'
            cons = format_item(id2item[b])
            writer.writerow([t0, t1, ants, cons, supAB, f"{conf:.2f}", f"{lift:.2f}"])

    print(f"[Window {idx}] CSV update done.")

print("\n[Process] All windows processed. CSV files are ready.")

