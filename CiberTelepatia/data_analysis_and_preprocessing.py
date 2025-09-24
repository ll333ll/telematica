import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

def run_data_analysis():
    print("Cargando datos de entrenamiento...")
    train_url = 'https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt'
    try:
        df = pd.read_csv(train_url, header=None, names=column_names)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return

    print("Realizando preprocesamiento inicial y cálculo de importancia...")
    # Crear objetivo binario para detección de intrusiones
    df['is_attack'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)

    # Calcular la puntuación de importancia de llegada
    # Se añade 1 para evitar log(0) si src_bytes + dst_bytes es 0
    df['importance_score'] = np.log(df['src_bytes'] + df['dst_bytes'] + 1)

    print("Generando visualizaciones con Plotly...")

    # 1. Distribución de Tipos de Ataque
    attack_counts = df['attack'].value_counts().reset_index()
    attack_counts.columns = ['Attack Type', 'Count']
    fig = px.bar(attack_counts, x='Attack Type', y='Count', title='Distribución de Tipos de Ataque en NSL-KDD')
    fig.write_html("attack_type_distribution.html")
    print("Gráfico 'attack_type_distribution.html' generado.")

    # 2. Distribución de la Puntuación de Importancia
    fig = px.histogram(df, x='importance_score', nbins=50, title='Distribución de la Puntuación de Importancia de Llegada')
    fig.write_html("importance_score_distribution.html")
    print("Gráfico 'importance_score_distribution.html' generado.")

    # 3. Puntuación de Importancia vs. Tipo de Conexión (Normal vs. Ataque)
    fig = px.box(df, x='is_attack', y='importance_score', 
                 title='Puntuación de Importancia por Tipo de Conexión',
                 labels={'is_attack': 'Tipo de Conexión (0=Normal, 1=Ataque)'})
    fig.write_html("importance_vs_attack.html")
    print("Gráfico 'importance_vs_attack.html' generado.")

    # 4. Relación entre Duración y Bytes (con color por tipo de ataque)
    # Muestra solo un subconjunto para evitar sobrecarga visual si hay muchos puntos
    sample_df = df.sample(n=min(10000, len(df)), random_state=42) # Muestra hasta 10000 puntos
    fig = px.scatter(sample_df, x='duration', y='src_bytes', color='attack',
                     log_x=True, log_y=True, size_max=10, 
                     title='Duración vs. Bytes Enviados (Muestra) por Tipo de Ataque')
    fig.write_html("duration_vs_src_bytes_attack.html")
    print("Gráfico 'duration_vs_src_bytes_attack.html' generado.")

    print("Análisis de datos inicial completado. Archivos HTML generados en la carpeta CiberTelepatia.")

if __name__ == "__main__":
    run_data_analysis()
