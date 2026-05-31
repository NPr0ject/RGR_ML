"""
РГР: Web-приложение (дашборд) для инференса моделей ML и анализа данных
Тема: Классификация мошеннических транзакций

Структура:
  - Страница 1: Информация о разработчике
  - Страница 2: Информация о наборе данных
  - Страница 3: Визуализации зависимостей
  - Страница 4: Инференс моделей ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="РГР: ML Inference Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Навигация через sidebar
st.sidebar.title(" Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ["О разработчике", " О наборе данных", " Визуализации", " Инференс моделей"],
    index=0
)

@st.cache_resource
def load_models():
    """Загрузка сериализованных моделей и артефактов."""
    models = {}
    metrics = {}
    scaler = None
    feature_names = None

    models_dir = 'models'
    if not os.path.exists(models_dir):
        st.error(f"Папка '{models_dir}' не найдена! Сначала запустите train_and_serialize.py")
        return models, metrics, scaler, feature_names

    # Загрузка scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    # Загрузка имён признаков
    features_path = os.path.join(models_dir, 'feature_names.pkl')
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)

    # Загрузка метрик
    metrics_path = os.path.join(models_dir, 'metrics.pkl')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)

    # Загрузка моделей
    for fname in os.listdir(models_dir):
        if fname.endswith('.pkl') and fname not in ['scaler.pkl', 'feature_names.pkl', 'metrics.pkl']:
            name = fname.replace('.pkl', '')
            with open(os.path.join(models_dir, fname), 'rb') as f:
                models[name] = pickle.load(f)

    return models, metrics, scaler, feature_names


@st.cache_data
def load_dataset():
    """Загрузка датасета для визуализаций."""
    # Пробуем разные пути
    for path in ['new_data_card.csv', 'data/new_data_card.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
            return df
    return None


# ============================================================
# СТРАНИЦА 1: О разработчике
# ============================================================
if page == "👤 О разработчике":
    st.title("👤 Информация о разработчике")

    st.markdown("---")

    col_photo, col_info = st.columns([1, 2])

    with col_photo:
        photo_path = "developer_photo.jpg"
        st.image(photo_path, width=250)

    with col_info:
        st.markdown("### Иванов Марк Игоревич")
        st.markdown("**Группа:** МО-241")

        st.markdown("---")
        st.markdown("### РГР")
        st.markdown(
            """
            **«Разработка Web-приложения (дашборда) для инференса моделей ML
            и анализа данных»**

            Направление: Классификация мошеннических транзакций
            """
        )

    st.markdown("---")
    st.markdown("###  Задачи РГР")
    st.markdown(
        """
        1. Обучить и сериализовать 6 моделей машинного обучения для задачи классификации мошеннических транзакций.
        2. Разработать веб-приложение (дашборд) на базе Streamlit для инференса моделей и визуализации данных.
        3. Провести анализ данных и продемонстрировать прогнозирование на примерах корректных данных и данных с выбросами.
        """
    )


# ============================================================
# СТРАНИЦА 2: О наборе данных
# ============================================================
elif page == " О наборе данных":
    st.title(" Информация о наборе данных")

    df = load_dataset()

    st.markdown("### Описание предметной области")
    st.markdown(
        """
        Датасет содержит информацию о транзакциях по банковским картам. Цель — определить,
        является ли транзакция мошеннической (fraud = True) или легальной (fraud = False).

        Задача классификации мошеннических транзакций является критически важной для
        банковского сектора и систем электронных платежей. Мошеннические операции составляют
        небольшую долю от общего объёма транзакций, однако их своевременное обнаружение
         
        Дисбаланс классов (мошеннические транзакции составляют около 8-12% от общего числа)
        """
    )

    if df is not None:
        st.markdown("### Основные характеристики датасета")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Количество строк", f"{df.shape[0]:,}")
        with col2:
            st.metric("Количество признаков", f"{df.shape[1] - 1}")
        with col3:
            fraud_pct = df['fraud'].mean() * 100
            st.metric("Доля мошенничества", f"{fraud_pct:.1f}%")

        st.markdown("---")
        st.markdown("### Описание признаков")

        feature_descriptions = {
            'distance_from_home': 'Расстояние от дома до места совершения транзакции (км)',
            'distance_from_last_transaction': 'Расстояние от предыдущей транзакции (км)',
            'ratio_to_median_purchase_price': 'Отношение суммы транзакции к медианной сумме покупок',
            'repeat_retailer': 'Повторная транзакция в том же магазине (0/1)',
            'used_chip': 'Использование чипа карты (0/1)',
            'used_pin_number': 'Использование PIN-кода (0/1)',
            'online_order': 'Онлайн-заказ (0/1)',
            'fraud': 'Целевой признак: мошенническая транзакция (True/False)'
        }

        desc_data = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            desc = feature_descriptions.get(col, '—')
            n_unique = df[col].nunique()
            n_null = df[col].isnull().sum()
            desc_data.append({
                'Признак': col,
                'Тип': dtype,
                'Уникальных': n_unique,
                'Пропусков': n_null,
                'Описание': desc
            })

        st.dataframe(pd.DataFrame(desc_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Предобработка данных и EDA")
        st.markdown(
            """
            **Этапы предобработки:**

            1. **Удаление лишних столбцов**: столбец `Unnamed: 0` удалён как технический артефакт.
            2. **Масштабирование признаков**: применён `StandardScaler` для приведения признаков
               к нулевому среднему и единичной дисперсии, что необходимо для корректной работы
               моделей (Logistic Regression, kNN, SVM).
            3. **Балансировка классов**: применён `SMOTE` (Synthetic Minority Over-sampling
               Technique) на обучающей выборке для устранения дисбаланса классов. Доля
               мошеннических транзакций увеличена с ~8-12% до 50%.
            4. **Стратифицированное разделение**: данные разделены на обучающую (80%) и
               тестовую (20%) выборки с сохранением пропорции классов.

            **Ключевые находки EDA:**
            - Признак `ratio_to_median_purchase_price` имеет наибольшую положительную
              корреляцию с мошенничеством (коэффициент логистической регрессии ≈ 3.43).
            - Использование PIN-кода (`used_pin_number`) — сильный защитный фактор
              (коэффициент ≈ -3.06).
            - Онлайн-заказы (`online_order`) чаще ассоциируются с мошенничеством (коэффициент ≈ 2.40).
            """
        )

        st.markdown("---")
        st.markdown("### Пример данных")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    else:
        st.warning("Датасет не найден.")


# ============================================================
# СТРАНИЦА 3: Визуализации
# ============================================================
elif page == "Визуализации":
    st.title("Визуализации зависимостей в наборе данных")

    df = load_dataset()

    if df is not None:
        df['fraud'] = df['fraud'].astype(int)

        COLORS = ['#2ecc71', '#e74c3c']

        st.markdown("### 1. Распределение целевой переменной (fraud)")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            counts = df['fraud'].value_counts().sort_index()
            labels_bar = ['Легальная (0)', 'Мошенническая (1)']
            bar_colors = [COLORS[0] if i == 0 else COLORS[1] for i in counts.index]
            bars = ax.bar(labels_bar, counts.values, color=bar_colors,
                          edgecolor='black', linewidth=0.5)
            ax.set_ylabel('Количество транзакций', fontsize=12)
            ax.set_title('Распределение классов', fontsize=14, fontweight='bold')
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f'{val:,}', ha='center', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            fraud_pct = df['fraud'].mean() * 100
            pie_labels = ['Легальные', 'Мошеннические']
            sizes = [100 - fraud_pct, fraud_pct]
            explode = (0, 0.08)
            wedges, texts, autotexts = ax.pie(
                sizes, explode=explode, labels=pie_labels,
                colors=COLORS, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12}
            )
            ax.set_title('Доля мошеннических транзакций', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("**Вывод:** Наблюдается значительный дисбаланс классов — мошеннические "
                    "транзакции составляют лишь небольшую долю от общего объёма, что "
                    "обусловливает необходимость применения SMOTE.")

        st.markdown("---")

        # ---- Визуализация 2: Корреляционная матрица ----
        st.markdown("### 2. Корреляционная матрица признаков")

        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 11, 'fontweight': 'bold'}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_title('Корреляционная матрица', fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("**Вывод:** Признаки в целом слабо коррелируют между собой, что "
                    "подтверждает их независимость и полезность для модели. Признак "
                    "`ratio_to_median_purchase_price` показывает наибольшую корреляцию с `fraud`.")

        st.markdown("---")

        # ---- Визуализация 3: Box-plot ключевых признаков по классам ----
        st.markdown("### 3. Распределение признаков по классам (Box-plot)")
        key_features = ['distance_from_home', 'distance_from_last_transaction',
                        'ratio_to_median_purchase_price']

        selected_feature = st.selectbox(
            "Выберите признак:", key_features, index=2, key="boxplot_feature"
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x='fraud', y=selected_feature, data=df,
            palette=COLORS, ax=ax, hue='fraud', legend=False
        )
        ax.set_xlabel('Класс (0 — легальная, 1 — мошенническая)', fontsize=12)
        ax.set_ylabel(selected_feature, fontsize=12)
        ax.set_title(f'Распределение {selected_feature} по классам',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(f"**Вывод:** Мошеннические транзакции демонстрируют значительно "
                    f"большие значения `{selected_feature}`, что указывает на сильный "
                    f"разделительный потенциал данного признака.")

        st.markdown("---")

        # ---- Визуализация 4: Парные диаграммы (scatter + KDE) ----
        st.markdown("### 4. Парные диаграммы рассеяния")
        st.markdown("Ключевые признаки для выявления взаимосвязей и кластеризации классов. "
                    "Используется подвыборка для скорости отрисовки.")

        pair_features = ['ratio_to_median_purchase_price', 'distance_from_home',
                         'online_order', 'fraud']

        sample_size = min(3000, len(df))
        df_sample = df[pair_features].sample(n=sample_size, random_state=42)

        g = sns.pairplot(
            df_sample,
            hue='fraud',
            palette={0: COLORS[0], 1: COLORS[1]},
            diag_kind='kde',
            plot_kws={'alpha': 0.4, 's': 12},
            height=1.6
        )
        g.fig.suptitle('Парные диаграммы рассеяния', y=1.02,
                       fontsize=12, fontweight='bold')
        st.pyplot(g.fig, use_container_width=True)
        plt.close(g.fig)

        st.markdown("**Вывод:** Мошеннические транзакции формируют отдельный кластер "
                    "в пространстве ключевых признаков, что подтверждает возможность "
                    "эффективной классификации.")

        st.markdown("---")

        # ---- Визуализация 5: Гистограммы признаков ----
        st.markdown("### 5. Гистограммы распределения признаков")
        feature_for_hist = st.selectbox(
            "Выберите признак для гистограммы:",
            [col for col in df.columns if col != 'fraud'],
            index=2,
            key="hist_feature"
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        for label, color in zip([0, 1], COLORS):
            subset = df[df['fraud'] == label][feature_for_hist]
            lbl = 'Легальная' if label == 0 else 'Мошенническая'
            ax.hist(subset, bins=50, alpha=0.6, color=color,
                    label=lbl, edgecolor='black', linewidth=0.3)
        ax.set_xlabel(feature_for_hist, fontsize=12)
        ax.set_ylabel('Частота', fontsize=12)
        ax.set_title(f'Распределение {feature_for_hist}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:
        st.warning(" Датасет не найден. Загрузите `new_data_card.csv` в папку проекта.")


# ============================================================
# СТРАНИЦА 4: Инференс моделей
# ============================================================
elif page == "Инференс моделей":
    st.title("Инференс моделей машинного обучения")

    models, metrics, scaler, feature_names = load_models()

    if not models:
        st.error(" Модели не найдены! Сначала запустите `train_and_serialize.py`")
        st.stop()

    # ---- Таблица метрик ----
    st.markdown("### Сравнение моделей")
    if metrics:
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = 'Модель'
        st.dataframe(metrics_df.style.format('{:.4f}'), use_container_width=True)

        # Лучшая модель
        best_model_name = max(metrics.keys(), key=lambda k: metrics[k]['F1'])
        st.success(f" Лучшая модель по F1: **{best_model_name}** (F1 = {metrics[best_model_name]['F1']:.4f})")
    else:
        st.info("Метрики недоступны")

    st.markdown("---")

    # ---- Выбор модели ----
    st.markdown("### Выбор модели для инференса")
    model_names = list(models.keys())
    selected_model_name = st.selectbox(
        "Выберите модель:",
        model_names,
        index=0
    )
    selected_model = models[selected_model_name]

    st.markdown("---")

    # ---- Способ ввода данных ----
    st.markdown("### Ввод данных для прогнозирования")
    input_method = st.radio(
        "Способ ввода данных:",
        [" Ручной ввод признаков", " Загрузка CSV-файла"],
        horizontal=True
    )

    if feature_names is None:
        st.error("Имена признаков не загружены!")
        st.stop()

    feature_units = {
        'distance_from_home': 'км',
        'distance_from_last_transaction': 'км',
        'ratio_to_median_purchase_price': '(безразмерный)',
        'repeat_retailer': '(0 или 1)',
        'used_chip': '(0 или 1)',
        'used_pin_number': '(0 или 1)',
        'online_order': '(0 или 1)'
    }

    feature_hints = {
        'distance_from_home': 'Расстояние от дома (0 — дома, типичные значения 0–100)',
        'distance_from_last_transaction': 'Расстояние от последней транзакции (0 — рядом, типичные значения 0–50)',
        'ratio_to_median_purchase_price': 'Отношение к медианной цене (1 — средняя, >3 — подозрительно)',
        'repeat_retailer': 'Повторный магазин (0 — нет, 1 — да)',
        'used_chip': 'Использование чипа (0 — нет, 1 — да)',
        'used_pin_number': 'Использование PIN (0 — нет, 1 — да)',
        'online_order': 'Онлайн-заказ (0 — нет, 1 — да)'
    }

    if input_method == " Ручной ввод признаков":
        st.markdown("Введите значения признаков для транзакции:")

        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(feature_names):
            with cols[i % 2]:
                unit = feature_units.get(feat, '')
                hint = feature_hints.get(feat, '')
                if feat in ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']:
                    val = st.selectbox(
                        f"{feat} {unit}",
                        options=[0, 1],
                        format_func=lambda x: f"{x} ({'Да' if x == 1 else 'Нет'})",
                        key=f"input_{feat}"
                    )
                else:
                    default_val = 1.0 if 'ratio' in feat else 5.0
                    val = st.number_input(
                        f"{feat} {unit}",
                        min_value=0.0,
                        value=default_val,
                        step=0.1,
                        help=hint,
                        key=f"input_{feat}"
                    )
                input_data[feat] = val

        # Кнопка прогнозирования
        if st.button("🔍 Получить прогноз", type="primary", use_container_width=True):
            # Формируем DataFrame
            input_df = pd.DataFrame([input_data])

            # Валидация
            validation_errors = []
            for feat in ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']:
                if input_data[feat] not in [0, 1]:
                    validation_errors.append(f"{feat} должен быть 0 или 1")
            for feat in ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']:
                if input_data[feat] < 0:
                    validation_errors.append(f"{feat} не может быть отрицательным")

            if validation_errors:
                for err in validation_errors:
                    st.error(f" {err}")
            else:
                # Масштабирование
                if scaler:
                    input_scaled = scaler.transform(input_df)
                else:
                    input_scaled = input_df.values

                # Прогноз
                prediction = selected_model.predict(input_scaled)[0]
                try:
                    probability = selected_model.predict_proba(input_scaled)[0]
                except AttributeError:
                    probability = None

                # Вывод результата
                st.markdown("---")
                st.markdown("###  Результат прогнозирования")

                if prediction == 1:
                    st.error(" **Транзакция классифицирована как МОШЕННИЧЕСКАЯ**")
                    if probability is not None:
                        st.metric("Вероятность мошенничества", f"{probability[1]*100:.1f}%")
                else:
                    st.success(" **Транзакция классифицирована как ЛЕГАЛЬНАЯ**")
                    if probability is not None:
                        st.metric("Вероятность легальности", f"{probability[0]*100:.1f}%")

                # Детали
                with st.expander(" Подробные вероятности"):
                    if probability is not None:
                        prob_df = pd.DataFrame({
                            'Класс': ['Легальная (0)', 'Мошенническая (1)'],
                            'Вероятность': [f"{probability[0]*100:.2f}%", f"{probability[1]*100:.2f}%"]
                        })
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)

                with st.expander(" Введённые значения"):
                    display_df = pd.DataFrame([{
                        'Признак': k,
                        'Значение': v,
                        'Единица': feature_units.get(k, '—')
                    } for k, v in input_data.items()])
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ---- Загрузка CSV ----
    elif input_method == " Загрузка CSV-файла":
        st.markdown("Загрузите CSV-файл с транзакциями для пакетного прогнозирования.")
        st.info(f" Ожидаемые столбцы: {', '.join(feature_names)}")

        uploaded_file = st.file_uploader(
            "Выберите CSV-файл",
            type=['csv'],
            help="Файл должен содержать столбцы с именами признаков"
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)

                # Проверка наличия нужных столбцов
                missing_cols = set(feature_names) - set(uploaded_df.columns)
                if missing_cols:
                    st.error(f" В файле отсутствуют столбцы: {', '.join(missing_cols)}")
                else:
                    st.success(f" Файл загружен: {uploaded_df.shape[0]} транзакций")

                    # Показать данные
                    with st.expander("👀 Предпросмотр загруженных данных"):
                        st.dataframe(uploaded_df.head(10), use_container_width=True, hide_index=True)

                    # Прогнозирование
                    if st.button("🔍 Получить прогнозы для всех транзакций", type="primary"):
                        input_data = uploaded_df[feature_names]

                        # Масштабирование
                        if scaler:
                            input_scaled = scaler.transform(input_data)
                        else:
                            input_scaled = input_data.values

                        predictions = selected_model.predict(input_scaled)
                        try:
                            probabilities = selected_model.predict_proba(input_scaled)
                            uploaded_df['Probability_Fraud'] = probabilities[:, 1]
                        except AttributeError:
                            pass

                        uploaded_df['Prediction'] = predictions
                        uploaded_df['Prediction_Label'] = uploaded_df['Prediction'].map(
                            {0: ' Легальная', 1: ' Мошенническая'}
                        )

                        # Статистика
                        st.markdown("### Результаты прогнозирования")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total = len(predictions)
                            st.metric("Всего транзакций", total)
                        with col2:
                            n_fraud = predictions.sum()
                            st.metric("Мошеннических", int(n_fraud))
                        with col3:
                            fraud_rate = n_fraud / total * 100 if total > 0 else 0
                            st.metric("Доля мошенничества", f"{fraud_rate:.1f}%")

                        # Таблица результатов
                        st.dataframe(uploaded_df, use_container_width=True, hide_index=True)

                        # Скачать результаты
                        csv = uploaded_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            " Скачать результаты (CSV)",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f" Ошибка при обработке файла: {str(e)}")

    # ---- Тестовые примеры ----
    st.markdown("---")
    st.markdown("### Быстрые тестовые примеры")

    test_col1, test_col2 = st.columns(2)

    with test_col1:
        st.markdown("**Корректная легальная транзакция**")
        st.caption("Рядом с домом, используется PIN, обычная сумма")
        if st.button("Протестировать легальную", key="test_normal"):
            normal_data = {
                'distance_from_home': 2.5,
                'distance_from_last_transaction': 1.0,
                'ratio_to_median_purchase_price': 0.8,
                'repeat_retailer': 1,
                'used_chip': 1,
                'used_pin_number': 1,
                'online_order': 0
            }
            input_df = pd.DataFrame([normal_data])
            if scaler:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values
            pred = selected_model.predict(input_scaled)[0]
            try:
                prob = selected_model.predict_proba(input_scaled)[0]
            except:
                prob = None

            if pred == 1:
                st.error(f" Мошенническая (вероятность: {prob[1]*100:.1f}%)" if prob is not None else " Мошенническая")
            else:
                st.success(f" Легальная (вероятность: {prob[0]*100:.1f}%)" if prob is not None else " Легальная")

    with test_col2:
        st.markdown("**Подозрительная транзакция (с выбросами)**")
        st.caption("Далеко от дома, огромная сумма, онлайн, без PIN")
        if st.button("Протестировать подозрительную", key="test_fraud"):
            fraud_data = {
                'distance_from_home': 150.0,
                'distance_from_last_transaction': 80.0,
                'ratio_to_median_purchase_price': 8.5,
                'repeat_retailer': 0,
                'used_chip': 0,
                'used_pin_number': 0,
                'online_order': 1
            }
            input_df = pd.DataFrame([fraud_data])
            if scaler:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values
            pred = selected_model.predict(input_scaled)[0]
            try:
                prob = selected_model.predict_proba(input_scaled)[0]
            except:
                prob = None

            if pred == 1:
                st.error(f" Мошенническая (вероятность: {prob[1]*100:.1f}%)" if prob is not None else " Мошенническая")
            else:
                st.success(f" Легальная (вероятность: {prob[0]*100:.1f}%)" if prob is not None else " Легальная")


    st.markdown("---")
    st.markdown("### Сравнение всех моделей на одной транзакции")


    compare_cols = st.columns(4)
    with compare_cols[0]:
        c_distance_home = st.number_input("distance_from_home", min_value=0.0, value=25.0, step=1.0, key="cmp_dist_home")
    with compare_cols[1]:
        c_distance_last = st.number_input("distance_from_last_transaction", min_value=0.0, value=15.0, step=1.0, key="cmp_dist_last")
    with compare_cols[2]:
        c_ratio = st.number_input("ratio_to_median_purchase_price", min_value=0.0, value=3.2, step=0.1, key="cmp_ratio")
    with compare_cols[3]:
        c_repeat = st.selectbox("repeat_retailer", options=[0, 1], format_func=lambda x: f"{x} ({'Да' if x == 1 else 'Нет'})", key="cmp_repeat")

    compare_cols2 = st.columns(3)
    with compare_cols2[0]:
        c_chip = st.selectbox("used_chip", options=[0, 1], format_func=lambda x: f"{x} ({'Да' if x == 1 else 'Нет'})", key="cmp_chip")
    with compare_cols2[1]:
        c_pin = st.selectbox("used_pin_number", options=[0, 1], format_func=lambda x: f"{x} ({'Да' if x == 1 else 'Нет'})", key="cmp_pin")
    with compare_cols2[2]:
        c_online = st.selectbox("online_order", options=[0, 1], format_func=lambda x: f"{x} ({'Да' if x == 1 else 'Нет'})", key="cmp_online")

    if st.button(" Сравнить все модели", type="primary", use_container_width=True):
        compare_data = {
            'distance_from_home': c_distance_home,
            'distance_from_last_transaction': c_distance_last,
            'ratio_to_median_purchase_price': c_ratio,
            'repeat_retailer': c_repeat,
            'used_chip': c_chip,
            'used_pin_number': c_pin,
            'online_order': c_online
        }
        input_df = pd.DataFrame([compare_data])
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values

        results = []
        for model_name, model_obj in models.items():
            pred = model_obj.predict(input_scaled)[0]
            try:
                prob = model_obj.predict_proba(input_scaled)[0]
                prob_fraud = prob[1] * 100
                prob_legal = prob[0] * 100
            except:
                prob_fraud = None
                prob_legal = None

            results.append({
                'Модель': model_name,
                'Прогноз': ' Мошенническая' if pred == 1 else ' Легальная',
                'P(легальная)': f"{prob_legal:.1f}%" if prob_legal is not None else '—',
                'P(мошенничество)': f"{prob_fraud:.1f}%" if prob_fraud is not None else '—',
                '_pred': pred,
                '_prob_fraud': prob_fraud
            })

        results_df = pd.DataFrame(results)
        n_fraud = sum(1 for r in results if r['_pred'] == 1)
        n_legal = sum(1 for r in results if r['_pred'] == 0)

        # Таблица результатов
        display_df = results_df[['Модель', 'Прогноз', 'P(легальная)', 'P(мошенничество)']]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Визуализация — горизонтальная столбчатая диаграмма вероятностей
        prob_values = [r['_prob_fraud'] for r in results if r['_prob_fraud'] is not None]
        model_labels = [r['Модель'] for r in results if r['_prob_fraud'] is not None]
        if prob_values:
            fig, ax = plt.subplots(figsize=(10, max(3, len(prob_values) * 0.7)))
            bar_colors = ['#e74c3c' if p > 50 else '#2ecc71' for p in prob_values]
            bars = ax.barh(model_labels, prob_values, color=bar_colors, edgecolor='black', linewidth=0.5)
            ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, label='Порог 50%')
            ax.set_xlabel('Вероятность мошенничества (%)', fontsize=12)
            ax.set_title('Сравнение вероятностей мошенничества по моделям', fontsize=14, fontweight='bold')
            for bar, val in zip(bars, prob_values):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
            ax.set_xlim(0, 105)
            ax.legend(loc='best')
            plt.tight_layout()
            st.pyplot(fig)

        # Итоговый вердикт
        st.markdown("---")
        if n_fraud > 0 and n_legal > 0:
            st.warning(
                f" **Модели расходятся!**\n\n"
                f"-  Легальная: **{n_legal}** из {n_legal + n_fraud} моделей\n"
                f"-  Мошенническая: **{n_fraud}** из {n_legal + n_fraud} моделей\n\n"
                f"Это граничный случай: часть признаков указывает на легальную операцию "
                f"(PIN, повторный магазин), а часть — на мошенничество (онлайн, повышенная сумма, без чипа). "
                f"Разные алгоритмические семейства по-разному взвешивают эти факторы."
            )
        elif n_fraud == 0:
            st.success(f" Все {n_legal} моделей согласны: транзакция легальная")
        else:
            st.error(f" Все {n_fraud} моделей согласны: транзакция мошенническая")
