import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_ids_project():
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

    # --- 1. Cargar y Preprocesar Datos de Entrenamiento ---
    print("Cargando datos de entrenamiento...")
    train_url = 'https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt'
    try:
        train_df = pd.read_csv(train_url, header=None, names=column_names)
    except Exception as e:
        print(f"Error al cargar los datos de entrenamiento: {e}")
        return

    print("Preprocesando datos de entrenamiento...")
    # Crear objetivo binario: 1 si es ataque, 0 si es normal
    train_df['is_attack'] = train_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    train_df = train_df.drop(['attack', 'difficulty'], axis=1)

    # One-Hot Encoding para variables categóricas
    categorical_cols = ['protocol_type', 'service', 'flag']
    train_df_processed = pd.get_dummies(train_df, columns=categorical_cols)

    # Separar características (X) y objetivo (y)
    X_train = train_df_processed.drop('is_attack', axis=1)
    y_train = train_df_processed['is_attack']

    # --- 2. Entrenar el Modelo ---
    print("Entrenando el modelo RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Modelo entrenado.")

    # --- 3. Cargar y Preprocesar Datos de Prueba ---
    print("\nCargando datos de prueba...")
    test_url = 'https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt'
    try:
        test_df = pd.read_csv(test_url, header=None, names=column_names)
    except Exception as e:
        print(f"Error al cargar los datos de prueba: {e}")
        return

    print("Preprocesando datos de prueba...")
    test_df['is_attack'] = test_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df = test_df.drop(['attack', 'difficulty'], axis=1)
    test_df_processed = pd.get_dummies(test_df, columns=categorical_cols)

    # Alinear columnas: asegurar que el set de prueba tenga las mismas columnas que el de entrenamiento
    train_cols = X_train.columns
    test_cols = test_df_processed.columns
    missing_cols = set(train_cols) - set(test_cols)
    for c in missing_cols:
        test_df_processed[c] = 0
    test_df_processed = test_df_processed[train_cols] # Asegurar el mismo orden

    X_test = test_df_processed.drop('is_attack', axis=1)
    y_test = test_df_processed['is_attack']

    # --- 4. Evaluar el Modelo ---
    print("Evaluando el modelo...")
    predictions = model.predict(X_test)

    # Imprimir resultados
    print("\n--- Resultados de la Evaluación ---")
    accuracy = accuracy_score(y_test, predictions)
    print(f"Exactitud (Accuracy): {accuracy:.4f}")

    print("\nMatriz de Confusión:")
    # Filas: Realidad, Columnas: Predicción
    # [[Verdaderos Negativos (Normal), Falsos Positivos (Falsa Alarma)],
    #  [Falsos Negativos (Ataque no detectado), Verdaderos Positivos (Ataque detectado)]]
    print(confusion_matrix(y_test, predictions))

    print("\nReporte de Clasificación:")
    # Precision: De todo lo que predije como ataque, ¿qué porcentaje acerté?
    # Recall: De todos los ataques reales, ¿qué porcentaje detecté?
    # F1-score: Media armónica de precision y recall.
    print(classification_report(y_test, predictions, target_names=['Normal', 'Ataque']))

if __name__ == "__main__":
    run_ids_project()
