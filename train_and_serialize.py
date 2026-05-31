"""
Скрипт обучения и сериализации 6 моделей ML для РГР.
Датасет: new_data_card.csv (классификация мошеннических транзакций)
Целевая переменная: fraud (бинарная)

Модели:
  ML1 — Logistic Regression (классическая модель с учителем)
  ML2 — GradientBoostingClassifier (ансамбль - бустинг)
  ML3 — CatBoostClassifier (продвинутый градиентный бустинг)
  ML4 — RandomForestClassifier (ансамбль - бэггинг)
  ML5 — StackingClassifier (ансамбль - стэкинг)
  ML6 — (зарезервировано для полносвязной нейросети — пока пропускаем)
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print(" CatBoost не установлен. Модель ML3 будет заменена на AdaBoostClassifier.")

from sklearn.ensemble import AdaBoostClassifier


def load_and_preprocess(csv_path='new_data_card.csv', sample_size=None):
    """Загрузка и предобработка данных.
    
    Параметры:
        sample_size: если указан, берётся подвыборка для ускорения 
                     (рекомендуется 200000–500000 для быстрого прогона)
    """
    df = pd.read_csv(csv_path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    if sample_size and sample_size < len(df):
        print(f"Используется подвыборка: {sample_size} из {len(df)} строк")
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df['fraud'], random_state=42
        )

    X = df.drop(columns=['fraud'])
    y = df['fraud']

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   SMOTE: {len(X_train_scaled)} → ", end="", flush=True)
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"{len(X_train_bal)} строк")

    return X_train_bal, y_train_bal, X_test_scaled, y_test, scaler, feature_names


def train_all_models(X_train, y_train, X_test, y_test):
    """Обучение 6 моделей и сбор результатов."""

    models = {}
    metrics = {}

    print(" Обучение ML1: Logistic Regression...")
    lr = LogisticRegression(
        C=1.0, max_iter=2000, random_state=42, n_jobs=-1
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    models['ML1_LogisticRegression'] = lr
    metrics['ML1_LogisticRegression'] = {
        'F1': f1_score(y_test, y_pred_lr),
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'Precision': precision_score(y_test, y_pred_lr),
        'Recall': recall_score(y_test, y_pred_lr),
        'ROC_AUC': roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    }
    print(f"   F1 = {metrics['ML1_LogisticRegression']['F1']:.4f}")

    print("Обучение ML2: GradientBoostingClassifier...")
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    models['ML2_GradientBoosting'] = gb
    metrics['ML2_GradientBoosting'] = {
        'F1': f1_score(y_test, y_pred_gb),
        'Accuracy': accuracy_score(y_test, y_pred_gb),
        'Precision': precision_score(y_test, y_pred_gb),
        'Recall': recall_score(y_test, y_pred_gb),
        'ROC_AUC': roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
    }
    print(f"F1 = {metrics['ML2_GradientBoosting']['F1']:.4f}")

    print("Обучение ML3: CatBoostClassifier...")
    cb = CatBoostClassifier(
        iterations=150, learning_rate=0.1, depth=6,
        random_seed=42, verbose=False, thread_count=-1
    )
    cb.fit(X_train, y_train)
    y_pred_cb = cb.predict(X_test)
    models['ML3_CatBoost'] = cb
    metrics['ML3_CatBoost'] = {
        'F1': f1_score(y_test, y_pred_cb),
        'Accuracy': accuracy_score(y_test, y_pred_cb),
        'Precision': precision_score(y_test, y_pred_cb),
        'Recall': recall_score(y_test, y_pred_cb),
        'ROC_AUC': roc_auc_score(y_test, cb.predict_proba(X_test)[:, 1])
    }
    ml3_name = list(models.keys())[2]
    print(f"   F1 = {metrics[ml3_name]['F1']:.4f}")

    # ============================================================
    # ML4 — RandomForestClassifier (ансамбль - бэггинг)
    # ============================================================
    print(" Обучение ML4: RandomForestClassifier...")
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=None, min_samples_split=2,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    models['ML4_RandomForest'] = rf
    metrics['ML4_RandomForest'] = {
        'F1': f1_score(y_test, y_pred_rf),
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf),
        'Recall': recall_score(y_test, y_pred_rf),
        'ROC_AUC': roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    }
    print(f"   F1 = {metrics['ML4_RandomForest']['F1']:.4f}")

    # ============================================================
    # ML5 — StackingClassifier (ансамбль - стэкинг)
    # ============================================================
    print("Обучение ML5: StackingClassifier...")
    print("3 разнородных алгоритма + мета-классификатор")
    print("1) Logistic Regression — линейная модель")
    print("2) RandomForest — дерево/бэггинг")
    print("3) kNN — метрическая модель (instance-based)")

    base_learners = [
        ('lr', LogisticRegression(max_iter=1000, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ('knn', KNeighborsClassifier(n_neighbors=7, n_jobs=-1, algorithm='ball_tree'))
    ]
    meta_clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    stack = StackingClassifier(
        estimators=base_learners, final_estimator=meta_clf,
        cv=2, n_jobs=-1, passthrough=False
    )
    stack.fit(X_train, y_train)
    y_pred_stack = stack.predict(X_test)
    models['ML5_Stacking'] = stack
    metrics['ML5_Stacking'] = {
        'F1': f1_score(y_test, y_pred_stack),
        'Accuracy': accuracy_score(y_test, y_pred_stack),
        'Precision': precision_score(y_test, y_pred_stack),
        'Recall': recall_score(y_test, y_pred_stack),
        'ROC_AUC': roc_auc_score(y_test, stack.predict_proba(X_test)[:, 1])
    }
    print(f"   F1 = {metrics['ML5_Stacking']['F1']:.4f}")
    return models, metrics


def serialize_models(models, scaler, feature_names, metrics, output_dir='models'):
    """Сериализация моделей и артефактов."""
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем каждую модель
    for name, model in models.items():
        filepath = os.path.join(output_dir, f'{name}.pkl')

        # CatBoost — используем встроенный save_model
        if CATBOOST_AVAILABLE and 'CatBoost' in name:
            cb_path = os.path.join(output_dir, f'{name}.cbm')
            model.save_model(cb_path)
            print(f"{name} сохранена → {cb_path} (CatBoost native)")
            # Также сохраняем через pickle для единообразия
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)

        print(f"{name} сохранена → {filepath}")

    # Сохраняем scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler сохранён → {scaler_path}")

    # Сохраняем имена признаков
    features_path = os.path.join(output_dir, 'feature_names.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Feature names сохранены → {features_path}")

    # Сохраняем метрики
    metrics_path = os.path.join(output_dir, 'metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Metrics сохранены → {metrics_path}")

    return output_dir


def print_summary(metrics):
    """Вывод итоговой таблицы метрик."""
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА МЕТРИК МОДЕЛЕЙ")
    print("=" * 80)
    print(f"{'Модель':<30} {'F1':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'ROC_AUC':>8}")
    print("-" * 80)
    for name, m in metrics.items():
        print(f"{name:<30} {m['F1']:>8.4f} {m['Accuracy']:>10.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f} {m['ROC_AUC']:>8.4f}")
    print("=" * 80)
    best = max(metrics.keys(), key=lambda k: metrics[k]['F1'])
    print(f"\nЛучшая модель по F1: {best} (F1 = {metrics[best]['F1']:.4f})")


if __name__ == '__main__':
    DATA_PATH = 'new_data_card.csv'
    sample_size = 200000

    print("\n" + "=" * 60)
    print("РГР: Обучение и сериализация моделей ML")
    print("Датасет: Классификация мошеннических транзакций")
    print("=" * 60 + "\n")

    print(" Загрузка и предобработка данных...")
    X_train, y_train, X_test, y_test, scaler, feature_names = load_and_preprocess(DATA_PATH, sample_size=sample_size)
    print(f"Train: {X_train.shape[0]} объектов (после SMOTE)")
    print(f"Test:  {X_test.shape[0]} объектов")
    print(f"Признаки: {len(feature_names)} → {feature_names}")

    print("=" * 60 + "\n")
    print("Обучение моделей...")
    models, metrics = train_all_models(X_train, y_train, X_test, y_test)

    print("=" * 60 + "\n")
    print("Сериализация моделей...")
    serialize_models(models, scaler, feature_names, metrics, output_dir='models')

    print("=" * 60 + "\n")
    print_summary(metrics)
