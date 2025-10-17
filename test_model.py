import pandas as pd
import numpy as np
from joblib import load

def test_data_format():
    df = pd.read_csv("Dataa.csv")
    assert not df.empty, "Le dataset est vide"
    assert "Delivery_Time_min" in df.columns, "La variable cible est manquante"

def test_model_performance():
    # suppose que tu sauvegardes le modèle final dans best_model.joblib
    model = load("best_model.joblib")
    X = pd.read_csv("X_test.csv")
    y = pd.read_csv("y_test.csv").values.ravel()

    preds = model.predict(X)
    mae = np.mean(np.abs(preds - y))
    assert mae < 15, f"MAE trop élevée : {mae}"  # ajuste le seuil selon ton dataset
