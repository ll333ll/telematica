import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

# Nombres de las columnas para el dataset NSL-KDD
column_names = [
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

def load_and_preprocess_data(train_url, test_url, column_names):
    print("Cargando datos...")
    try:
        train_df = pd.read_csv(train_url, header=None, names=column_names)
        test_df = pd.read_csv(test_url, header=None, names=column_names)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None, None

    print("Preprocesando datos...")
    # Crear objetivo binario para detección de intrusiones
    train_df['is_attack'] = train_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['is_attack'] = test_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)

    # Calcular la puntuación de importancia de llegada
    train_df['importance_score'] = np.log(train_df['src_bytes'] + train_df['dst_bytes'] + 1)
    test_df['importance_score'] = np.log(test_df['src_bytes'] + test_df['dst_bytes'] + 1)

    # Eliminar columnas originales de ataque y dificultad
    train_df = train_df.drop(['attack', 'difficulty'], axis=1)
    test_df = test_df.drop(['attack', 'difficulty'], axis=1)

    # One-Hot Encoding para variables categóricas
    categorical_cols = ['protocol_type', 'service', 'flag']
    train_df_processed = pd.get_dummies(train_df, columns=categorical_cols)
    test_df_processed = pd.get_dummies(test_df, columns=categorical_cols)

    # Alinear columnas después del one-hot encoding
    # Esto es crucial para que los sets de entrenamiento y prueba tengan las mismas columnas
    common_cols = list(set(train_df_processed.columns) & set(test_df_processed.columns))
    train_df_processed = train_df_processed[common_cols]
    test_df_processed = test_df_processed[common_cols]

    # Asegurar que todas las columnas de entrenamiento estén en el set de prueba (rellenar con 0 si falta)
    missing_in_test = set(train_df_processed.columns) - set(test_df_processed.columns)
    for c in missing_in_test:
        test_df_processed[c] = 0
    test_df_processed = test_df_processed[train_df_processed.columns] # Asegurar el mismo orden

    # Separar características (X) y objetivos (y)
    X_train_ids = train_df_processed.drop(['is_attack', 'importance_score'], axis=1)
    y_train_ids = train_df_processed['is_attack']
    X_test_ids = test_df_processed.drop(['is_attack', 'importance_score'], axis=1)
    y_test_ids = test_df_processed['is_attack']

    X_train_importance = train_df_processed.drop(['is_attack', 'importance_score'], axis=1)
    y_train_importance = train_df_processed['importance_score']
    X_test_importance = test_df_processed.drop(['is_attack', 'importance_score'], axis=1)
    y_test_importance = test_df_processed['importance_score']

    # Escalar características numéricas (importante para algunos modelos como Regresión Logística, SVM)
    # Excluir columnas binarias creadas por get_dummies
    numerical_cols = X_train_ids.select_dtypes(include=np.number).columns.tolist()
    # Filtrar columnas que son resultado de one-hot encoding (ya son 0/1)
    # Una forma simple es excluir las que tienen solo 0 y 1 y un nombre que sugiere one-hot
    # Esto es una heurística, para un caso real se necesitaría una lista más precisa
    numerical_cols_filtered = [col for col in numerical_cols if not (X_train_ids[col].isin([0, 1]).all() and (col.startswith('protocol_type_') or col.startswith('service_') or col.startswith('flag_')))]

    scaler = StandardScaler()
    X_train_ids[numerical_cols_filtered] = scaler.fit_transform(X_train_ids[numerical_cols_filtered])
    X_test_ids[numerical_cols_filtered] = scaler.transform(X_test_ids[numerical_cols_filtered])
    X_train_importance[numerical_cols_filtered] = scaler.fit_transform(X_train_importance[numerical_cols_filtered])
    X_test_importance[numerical_cols_filtered] = scaler.transform(X_test_importance[numerical_cols_filtered])

    return X_train_ids, y_train_ids, X_test_ids, y_test_ids, X_train_importance, y_train_importance, X_test_importance, y_test_importance

def train_and_evaluate_classification(X_train, y_train, X_test, y_test):
    print("\n--- Detección de Intrusiones (Clasificación) ---")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        # 'SVC': SVC(random_state=42) # SVC es muy lento con datasets grandes, se comenta para agilizar
    }

    best_classifier_name = None
    best_classifier_score = -1
    best_classifier_model = None

    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"  {name} - Exactitud: {accuracy:.4f}")
        print(f"  {name} - Reporte de Clasificación:\n{classification_report(y_test, predictions, target_names=['Normal', 'Ataque'])}")
        
        if accuracy > best_classifier_score:
            best_classifier_score = accuracy
            best_classifier_name = name
            best_classifier_model = model
    
    print(f"\nMejor clasificador: {best_classifier_name} con Exactitud: {best_classifier_score:.4f}")

    # Optimización de hiperparámetros para el mejor clasificador (Random Forest)
    if best_classifier_name == 'Random Forest':
        print("\nOptimizando hiperparámetros para Random Forest (GridSearchCV)...")
        param_grid = {
            'n_estimators': [50, 100], # Reducido para agilizar la ejecución
            'max_depth': [None, 10] # Reducido para agilizar la ejecución
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=2, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Mejores parámetros: {grid_search.best_params_}")
        best_rf_model = grid_search.best_estimator_
        predictions_optimized = best_rf_model.predict(X_test)
        accuracy_optimized = accuracy_score(y_test, predictions_optimized)
        print(f"Exactitud (Random Forest Optimizado): {accuracy_optimized:.4f}")
        print(f"Reporte de Clasificación (Random Forest Optimizado):\n{classification_report(y_test, predictions_optimized, target_names=['Normal', 'Ataque'])}")

def train_and_evaluate_regression(X_train, y_train, X_test, y_test):
    print("\n--- Predicción de Importancia de Llegada (Regresión) ---")
    models = {
        'Linear Regression': LinearRegression(n_jobs=-1),
        'Random Forest Regressor': RandomForestRegressor(random_state=42, n_jobs=-1)
    }

    best_regressor_name = None
    best_regressor_score = -float('inf') # Usar R2 score, que puede ser negativo
    best_regressor_model = None

    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"  {name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
        
        if r2 > best_regressor_score:
            best_regressor_score = r2
            best_regressor_name = name
            best_regressor_model = model
    
    print(f"\nMejor regresor: {best_regressor_name} con R2 Score: {best_regressor_score:.4f}")

    # Optimización de hiperparámetros para el mejor regresor (Random Forest Regressor)
    if best_regressor_name == 'Random Forest Regressor':
        print("\nOptimizando hiperparámetros para Random Forest Regressor (GridSearchCV)...")
        param_grid = {
            'n_estimators': [50, 100], # Reducido para agilizar la ejecución
            'max_depth': [None, 10] # Reducido para agilizar la ejecución
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_grid, cv=2, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Mejores parámetros: {grid_search.best_params_}")
        best_rf_regressor_model = grid_search.best_estimator_
        predictions_optimized = best_rf_regressor_model.predict(X_test)
        mse_optimized = mean_squared_error(y_test, predictions_optimized)
        r2_optimized = r2_score(y_test, predictions_optimized)
        print(f"MSE (Random Forest Regressor Optimizado): {mse_optimized:.4f}, R2 Score: {r2_optimized:.4f}")

if __name__ == "__main__":
    train_url = 'https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt'
    test_url = 'https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt'

    X_train_ids, y_train_ids, X_test_ids, y_test_ids, \
    X_train_importance, y_train_importance, X_test_importance, y_test_importance = \
        load_and_preprocess_data(train_url, test_url, column_names)

    if X_train_ids is not None:
        train_and_evaluate_classification(X_train_ids, y_train_ids, X_test_ids, y_test_ids)
        train_and_evaluate_regression(X_train_importance, y_train_importance, X_test_importance, y_test_importance)
