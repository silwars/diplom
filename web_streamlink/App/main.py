import streamlit as st
import subprocess
import time
from datetime import timedelta
import os
import pandas as pd
from glob import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

from ids_utils import BINNING_SPEC, SELECTED_COLS, FEATURE_GROUPS, discretize_columns, row_features
from test_ids import run_test_ids
from hyperparameter_tuning import run_grid_search


# --- Директории проекта ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = BASE_DIR
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

# --- Настройка страницы ---
st.set_page_config(page_title="IDS Prototype UI", layout="wide")

# --- Вспомогательные функции ---
def run_script(script_name):
    """Запускает скрипт из APP_DIR и возвращает (success, stdout, stderr, elapsed)"""
    path = os.path.join(APP_DIR, script_name)
    start = time.time()
    proc = subprocess.run(["python", path], capture_output=True, text=True)
    elapsed = time.time() - start
    return proc.returncode == 0, proc.stdout, proc.stderr, elapsed


def find_latest_file(pattern):
    files = glob(os.path.join(DATA_DIR, pattern))
    return max(files, key=os.path.getmtime) if files else None


def list_files(pattern):
    files = glob(os.path.join(DATA_DIR, pattern))
    return sorted(files, key=os.path.getmtime, reverse=True)

# Инициализируем session_state
if "page" not in st.session_state:
    st.session_state.page = "Тест IDS"

# Сторонний код (импорты, утилиты)...

# Навигация в сайдбаре
with st.sidebar:
    st.markdown("## Навигация")
    st.write("")
    if st.button("Тест IDS"):
        st.session_state.page = "Тест IDS"
    st.write("")
    if st.button("Ассоциативные правила"):
        st.session_state.page = "Ассоциативные правила"
    st.write("")
    if st.button("Вычисление гиперпараметров"):
        st.session_state.page = "Вычисление гиперпараметров"

# **Здесь** присваиваем переменной page текущее значение:
page = st.session_state.page

# --- Страница Test IDS ---
if page == "Тест IDS":
    st.header("Тест IDS")

    # 1) Файловый селект
    unlabeled_files = sorted(glob(os.path.join(DATA_DIR, "*sampled2.csv")))
    if unlabeled_files:
        selected_unl = st.selectbox("Неразмеченный датасет",unlabeled_files,format_func=lambda f: os.path.basename(f),key="unl_ds")
        df_unl = pd.read_csv(selected_unl, parse_dates=["Timestamp"])
        st.subheader(f"Неразмеченный датасет ({os.path.basename(selected_unl)})")
        st.dataframe(df_unl)
    else:
        st.warning("В папке data/ не найден ни один файл с суффиксом _v4")

    ds_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
    selected_ds = st.selectbox("Размеченный датасет", ds_files, format_func=lambda f: os.path.basename(f))

    tmpl_files = sorted(glob(os.path.join(DATA_DIR, "generated_attack_templates*.csv")))
    selected_tmpl = st.selectbox("Шаблон атак", tmpl_files, format_func=lambda f: os.path.basename(f))

    rules_files = sorted(glob(os.path.join(DATA_DIR, "association_rules*.csv")))
    selected_rules = st.selectbox("Ассоц. правила", rules_files, format_func=lambda f: os.path.basename(f))

    # 2) Параметры порогов
    min_len_thresh = st.number_input("Минимальная длина A (len(A) ≥ …)",min_value=1, value=4, step=1,key="param_min_len")
    min_rules_thresh = st.number_input("Порог fires (MIN_RULES ≥ …)",min_value=0, value=150, step=10,key="param_min_rules")

    # 3) Запуск теста
    if st.button("Запустить тест"):
        with st.spinner("Выполняется Тест IDS..."):
            result = run_test_ids(dataset_path=selected_ds,template_path=selected_tmpl,rules_path=selected_rules,min_rules_threshold=min_rules_thresh,min_len_thresh=min_len_thresh)

            df_pred = result['predictions']

        # 4) Показ метрик
        m = result['metrics']
        st.metric("Precision", f"{m['precision']:.2%}")
        st.metric("Recall",    f"{m['recall']:.2%}")
        st.write(f"TP={m['TP']}, FP={m['FP']}, TN={m['TN']}, FN={m['FN']}")


        # 5) Предсказания
        st.subheader("Предсказания")
        st.dataframe(result['predictions'])
        # Подготовка arrays для метрик и графиков
        if {'Label','Predicted','fires'}.issubset(df_pred.columns):
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
            import matplotlib.pyplot as plt

            y_true  = (df_pred['Label']     == 'DDoS').astype(int).values
            y_pred  = (df_pred['Predicted'] == 'DDoS').astype(int).values
            y_score = df_pred['fires'].values

            #Графики
            # 1) ROC-кривая
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots(figsize=(5, 3))
            ax_roc.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
            ax_roc.plot([0,1], [0,1], linestyle='--')
            ax_roc.set_title('ROC кривая')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc='lower right')

            st.pyplot(fig_roc, use_container_width=False)

            # 2) Precision–Recall кривая
            precision, recall, _ = precision_recall_curve(y_true, y_score)

            fig_pr, ax_pr = plt.subplots(figsize=(5, 3))
            ax_pr.plot(recall, precision, marker='.')
            ax_pr.set_title('Precision–Recall кривая')
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')

            st.pyplot(fig_pr, use_container_width=False)


            # 3) Матрица ошибок
            cm = confusion_matrix(y_true, y_pred)

            fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
            cax = ax_cm.matshow(cm, cmap='Blues')
            fig_cm.colorbar(cax)
            ax_cm.set_xticks([0,1])
            ax_cm.set_yticks([0,1])
            ax_cm.set_xticklabels(['BENIGN','DDoS'])
            ax_cm.set_yticklabels(['BENIGN','DDoS'])
            ax_cm.xaxis.set_ticks_position('bottom')
            ax_cm.xaxis.tick_bottom()
            ax_cm.xaxis.set_label_position('bottom')
            ax_cm.set_xlabel('Предсказание')
            ax_cm.set_ylabel('Истина')
            ax_cm.set_title('Матрица ошибок')

            st.pyplot(fig_cm, use_container_width=False)


# --- Страница Associative Rules ---
elif page == "Ассоциативные правила":
    st.header("Генерация ассоциативных правил")
    if st.button("Создать правила", key="create_rules"):
        with st.spinner("Генерация правил..."):
            ok, out, err, elapsed = run_script("association_rules.py")
        if not ok:
            st.error(f"Ошибка создания правил:\n{err}")
        else:
            st.success(f"Правила сгенерированы за {elapsed:.2f} сек.")

    rules_file = find_latest_file("association_rules_no_label*.csv")
    if rules_file:
        df_rules = pd.read_csv(rules_file)
        st.subheader(f"Ассоциативные правила ({os.path.basename(rules_file)})")
        st.dataframe(df_rules)
    else:
        st.info("Правила ещё не созданы.")


# --- Страница Hyperparameter Tuning ---
elif page == "Вычисление гиперпараметров":
    st.header("Вычисление гиперпараметров")

    # 1) Выбор файлов
    ds_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
    selected_ds = st.selectbox(
    "1. Размеченный датасет",
    ds_files,
    format_func=lambda full: os.path.basename(full),
    key="ht_ds"
)

    tmpl_files = sorted(glob(os.path.join(DATA_DIR, "generated_attack_templates*.csv")))
    selected_tmpl = st.selectbox(
    "2. Шаблон атак",
    tmpl_files,
    format_func=lambda full: os.path.basename(full),
    key="ht_tmpl"
)

    rules_files = sorted(glob(os.path.join(DATA_DIR, "association_rules*.csv")))
    selected_rules = st.selectbox(
    "3. Ассоциативные правила",
    rules_files,
    format_func=lambda full: os.path.basename(full),
    key="ht_rules"
)

    # 2) Диапазоны параметров
    min_len_list = st.multiselect(
        "min_len (длина A)", [2,3,4,5],
        default=[2,3,4,5], key="ht_ml"
    )
    min_rules_list = st.multiselect(
        "min_rules (порог fires)", [50,100,150,200,250],
        default=[50,100,150,200,250], key="ht_mr"
    )

    # 3) Запуск grid search
    if st.button("Запустить Grid Search", key="run_grid"):
        with st.spinner("Выполняется grid search…"):
            df_grid = run_grid_search(
                dataset_path=selected_ds,
                template_path=selected_tmpl,
                rules_path=selected_rules,
                min_len_list=min_len_list,
                min_rules_list=min_rules_list
            )

        st.subheader("Результаты Grid Search")
        st.dataframe(df_grid, use_container_width=True)

        st.subheader("Heatmap F1-score")
        pivot = df_grid.pivot(index="min_len", columns="min_rules", values="f1")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax
        )
        ax.set_xlabel("min_rules")
        ax.set_ylabel("min_len")
        st.pyplot(fig, use_container_width=False)

