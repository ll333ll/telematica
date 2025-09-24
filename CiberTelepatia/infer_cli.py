"""
CLI de inferencia para el mejor modelo guardado (best_model.joblib).
Uso:
  python3 infer_cli.py --model outputs/best_model.joblib --csv path/a/nuevo.csv

El CSV debe tener las mismas columnas de entrenamiento después del preprocesado
(usa la misma canalización que run_ids_ml.py si quieres reproducir pasos).
"""
import argparse
import joblib
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='outputs/best_model.joblib')
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()

    clf = joblib.load(args.model)
    df = pd.read_csv(args.csv)
    preds = clf.predict(df)
    pd.DataFrame({'prediction': preds}).to_csv('predictions.csv', index=False)
    print('Predicciones guardadas en predictions.csv')

if __name__ == '__main__':
    main()

