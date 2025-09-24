import os
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Dict
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.utils import Bunch

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Modelos externos solicitados
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


TRAIN_URL = 'https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt'
TEST_URL  = 'https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt'

COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'difficulty'
]


def ensure_outputs_dir() -> Path:
    """Crea la carpeta de salidas y subcarpetas útiles (métricas y matrices)."""
    out = Path(__file__).resolve().parent / 'outputs'
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    (out / 'confusion_matrices').mkdir(parents=True, exist_ok=True)
    (out / 'plots').mkdir(parents=True, exist_ok=True)
    return out


def load_data(train_url: str = TRAIN_URL, test_url: str = TEST_URL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_url, header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_url, header=None, names=COLUMN_NAMES)
    return train_df, test_df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Bunch:
    # Etiqueta binaria
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['is_attack'] = (train_df['attack'] != 'normal').astype(int)
    test_df['is_attack'] = (test_df['attack'] != 'normal').astype(int)

    # Drop columnas no usadas
    train_df.drop(columns=['attack', 'difficulty'], inplace=True)
    test_df.drop(columns=['attack', 'difficulty'], inplace=True)

    # One-hot de categóricas y alineación de columnas
    cat_cols = ['protocol_type', 'service', 'flag']
    train_proc = pd.get_dummies(train_df, columns=cat_cols)
    test_proc = pd.get_dummies(test_df, columns=cat_cols)

    # Alinear columnas
    missing_cols = set(train_proc.columns) - set(test_proc.columns)
    for c in missing_cols:
        test_proc[c] = 0
    extra_cols = set(test_proc.columns) - set(train_proc.columns)
    for c in extra_cols:
        test_proc.drop(columns=[c], inplace=True)
    test_proc = test_proc[train_proc.columns]

    # Separar X, y
    y_train = train_proc['is_attack']
    X_train = train_proc.drop(columns=['is_attack'])
    y_test = test_proc['is_attack']
    X_test = test_proc.drop(columns=['is_attack'])

    # Escalado de columnas numéricas (evitar dummies)
    num_cols = [c for c in X_train.columns if not (c.startswith('protocol_type_') or c.startswith('service_') or c.startswith('flag_'))]
    scaler = StandardScaler(with_mean=False)  # sparse-friendly si aparece
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return Bunch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_cols=num_cols)


def _slug(name: str) -> str:
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')


def _save_per_model_outputs(outdir: Path, name: str, y_true, y_pred, y_score=None):
    """Guarda reporte de clasificación, matriz de confusión (CSV) y heatmap/curvas."""
    from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
    import matplotlib.pyplot as plt
    import seaborn as sns
    slug = _slug(name)
    rep = classification_report(y_true, y_pred, target_names=['Normal', 'Ataque'])
    (outdir / 'metrics' / f'{slug}_classification_report.txt').write_text(rep, encoding='utf-8')
    cm = confusion_matrix(y_true, y_pred)
    # Guardar matriz en CSV sencillo
    pd.DataFrame(cm, index=['Real_Normal','Real_Ataque'], columns=['Pred_Normal','Pred_Ataque']) \
        .to_csv(outdir / 'confusion_matrices' / f'{slug}_cm.csv')
    # Heatmap
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred_Normal','Pred_Ataque'], yticklabels=['Real_Normal','Real_Ataque'])
    plt.tight_layout()
    plt.savefig(outdir / 'plots' / f'{slug}_cm_heatmap.png')
    plt.close()
    # Curvas ROC/PR si hay score
    if y_score is not None:
        try:
            RocCurveDisplay.from_predictions(y_true, y_score)
            plt.tight_layout(); plt.savefig(outdir / 'plots' / f'{slug}_roc.png'); plt.close()
            PrecisionRecallDisplay.from_predictions(y_true, y_score)
            plt.tight_layout(); plt.savefig(outdir / 'plots' / f'{slug}_pr.png'); plt.close()
        except Exception:
            pass


def evaluate_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Entrena 10+ modelos (incluye CatBoost y LightGBM), y 3 ensambles.
    Devuelve un DataFrame con métricas y guarda reportes y matrices por modelo.
    """
    results = []

    # Modelos base (sin SVC RBF por costo computacional)
    models: Dict[str, object] = {
        'LogReg': LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1, random_state=42, class_weight='balanced'),
        'LinearSVC': LinearSVC(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'GaussianNB': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
    }

    if LGBMClassifier is not None:
        models['LightGBM'] = LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1)
    if CatBoostClassifier is not None:
        # silencioso para no saturar logs
        models['CatBoost'] = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, verbose=False, random_seed=42)

    # Entrenar y evaluar modelos base
    for name, model in models.items():
        print(f"Entrenando {name}…")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

        # ROC AUC si es posible con decision_function o predict_proba
        try:
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
            else:
                y_score = None
            roc = roc_auc_score(y_test, y_score) if y_score is not None else np.nan
        except Exception:
            roc = np.nan

        _save_per_model_outputs(ensure_outputs_dir(), name, y_test, y_pred, y_score)
        print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} ROC-AUC={roc if not np.isnan(roc) else float('nan'):.4f}")
        results.append({'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc})

    return pd.DataFrame(results).sort_values(by=['f1', 'accuracy'], ascending=False)


def evaluate_ensembles(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Entrena ensambles Voting, Stacking y Bagging, y guarda reportes/matrices."""
    results = []
    print("Entrenando ensambles…")

    voting = VotingClassifier(
        estimators=[('lr', LogisticRegression(max_iter=1000, solver='saga', random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
                    ('et', ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1))],
        voting='hard', n_jobs=-1
    )
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    _save_per_model_outputs(ensure_outputs_dir(), 'Voting(hard)', y_test, y_pred)
    results.append({'model': 'Voting(hard)', 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': np.nan})

    stacking = StackingClassifier(
        estimators=[('rf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
                    ('gb', GradientBoostingClassifier(random_state=42)),
                    ('lda', LinearDiscriminantAnalysis())],
        final_estimator=LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
        n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    _save_per_model_outputs(ensure_outputs_dir(), 'Stacking', y_test, y_pred)
    results.append({'model': 'Stacking', 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': np.nan})

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100, random_state=42, n_jobs=-1
    )
    bagging.fit(X_train, y_train)
    y_pred = bagging.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    _save_per_model_outputs(ensure_outputs_dir(), 'Bagging(Tree)', y_test, y_pred)
    results.append({'model': 'Bagging(Tree)', 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': np.nan})

    return pd.DataFrame(results).sort_values(by=['f1', 'accuracy'], ascending=False)


def tune_top_models(results_df: pd.DataFrame, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Hace RandomizedSearchCV sobre los 3 mejores modelos por F1."""
    top = results_df.head(3)['model'].tolist()
    tuned = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for name in top:
        if name == 'RandomForest':
            base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif name == 'LogReg':
            base = LogisticRegression(max_iter=2000, solver='saga', n_jobs=-1, class_weight='balanced', random_state=42)
            params = {
                'C': np.logspace(-3, 1, 10)
            }
        elif name == 'ExtraTrees':
            base = ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            params = {
                'n_estimators': [200, 300, 500],
                'max_depth': [None, 10, 20]
            }
        else:
            continue
        print(f"Tuning {name}…")
        rs = RandomizedSearchCV(base, params, n_iter=8, scoring='f1', cv=cv, random_state=42, n_jobs=-1, verbose=0)
        rs.fit(X_train, y_train)
        best = rs.best_estimator_
        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        tuned.append({'model': f'{name}(tuned)', 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': np.nan})
    return pd.DataFrame(tuned)


def main():
    parser = argparse.ArgumentParser(description='IDS ML pipeline (NSL-KDD)')
    parser.add_argument('--phase', choices=['all', 'base', 'ensembles'], default='all',
                        help='Selecciona fase a ejecutar: modelos base, ensambles o todo')
    args = parser.parse_args()

    outdir = ensure_outputs_dir()
    # Semillas globales
    np.random.seed(42)
    random.seed(42)
    print("Cargando datos…")
    train_df, test_df = load_data()

    print("Preprocesando…")
    data = preprocess(train_df, test_df)

    final = []
    if args.phase in ('all', 'base'):
        print("Entrenando y evaluando modelos base…")
        base_res = evaluate_models(data.X_train, data.y_train, data.X_test, data.y_test)
        base_res.to_csv(outdir / 'results_base.csv', index=False)
        final.append(base_res)
    if args.phase in ('all', 'ensembles'):
        ens_res = evaluate_ensembles(data.X_train, data.y_train, data.X_test, data.y_test)
        ens_res.to_csv(outdir / 'results_ensembles.csv', index=False)
        final.append(ens_res)

    if final:
        results = pd.concat(final, ignore_index=True).sort_values(by=['f1', 'accuracy'], ascending=False)
        print("\nResultados ordenados por F1 y Accuracy:\n", results)
        results.to_csv(outdir / 'results_summary.csv', index=False)
        print(f"\nResumen guardado en: {outdir / 'results_summary.csv'}")

        # Tuning top-3
        tuned = tune_top_models(results, data.X_train, data.y_train, data.X_test, data.y_test)
        if not tuned.empty:
            tuned.to_csv(outdir / 'results_tuned.csv', index=False)
            results = pd.concat([results, tuned], ignore_index=True).sort_values(by=['f1', 'accuracy'], ascending=False)

        # Guardar resumen MD y mejor modelo
        top5 = results.head(5)
        md_lines = ["# Resumen de Resultados (Top-5)", "", "| Modelo | F1 | Accuracy |", "|---|---:|---:|"]
        for _, row in top5.iterrows():
            md_lines.append(f"| {row['model']} | {row['f1']:.4f} | {row['accuracy']:.4f} |")
        (outdir / 'results_summary.md').write_text("\n".join(md_lines), encoding='utf-8')

        # Persistir mejor modelo si es uno de los que tenemos instancia
        best_name = results.iloc[0]['model']
        best = None
        # entrenar de nuevo best para persistir con todos los datos de train
        if best_name.startswith('RandomForest'):
            best = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
        elif best_name.startswith('LogReg'):
            best = LogisticRegression(max_iter=2000, solver='saga', n_jobs=-1, class_weight='balanced', random_state=42)
        elif best_name.startswith('ExtraTrees'):
            best = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')
        if best is not None:
            import joblib
            best.fit(data.X_train, data.y_train)
            joblib.dump(best, outdir / 'best_model.joblib')
            print(f"Mejor modelo persistido en {outdir / 'best_model.joblib'}")


if __name__ == '__main__':
    main()
