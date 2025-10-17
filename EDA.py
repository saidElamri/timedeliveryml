import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# ==============================
# Charger et nettoyer les données
# ==============================
data_path = "Dataa.csv"
if not pd.io.common.file_exists(data_path):
    raise FileNotFoundError(f"Le fichier {data_path} est introuvable.")

df = pd.read_csv(data_path)

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Remove outliers (IQR method)
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Encode categorical variables
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ==============================
# Feature selection
# ==============================
target_col = 'Delivery_Time_min'
X = df.drop(target_col, axis=1)
y = df[target_col]

selector = SelectKBest(score_func=f_regression, k=4)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features], y, test_size=0.2, random_state=42
)

# ==============================
# Modèles RF + SVR
# ==============================
def train_rf_svr(X_train, X_test, y_train, y_test):
    mlflow.set_experiment("DeliveryTimePrediction")

    # Random Forest
    with mlflow.start_run(run_name="RandomForest_Model"):
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(random_state=42))
        ])
        rf_param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
        }
        rf_grid = GridSearchCV(
            rf_pipeline, param_grid=rf_param_grid, cv=5,
            scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0
        )
        rf_grid.fit(X_train, y_train)
        rf_best_model = rf_grid.best_estimator_
        rf_pred = rf_best_model.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)

    # SVR
    with mlflow.start_run(run_name="SVR_Model"):
        svr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR())
        ])
        svr_param_grid = {
            'model__kernel': ['rbf', 'linear'],
            'model__C': [0.1, 1, 10],
            'model__epsilon': [0.1, 0.5, 1]
        }
        svr_grid = GridSearchCV(
            svr_pipeline, param_grid=svr_param_grid, cv=5,
            scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0
        )
        svr_grid.fit(X_train, y_train)
        svr_best_model = svr_grid.best_estimator_
        svr_pred = svr_best_model.predict(X_test)
        svr_mae = mean_absolute_error(y_test, svr_pred)

    # Choisir le meilleur modèle
    if rf_mae < svr_mae:
        best_model = rf_best_model
    else:
        best_model = svr_best_model

    return best_model

best_model = train_rf_svr(X_train, X_test, y_train, y_test)

# ==============================
# Full pipeline
# ==============================
num_cols = X[selected_features].select_dtypes(include=np.number).columns.tolist()
cat_cols = X[selected_features].select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

final_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_regression, k=4)),
    ('model', best_model)
])

final_pipeline.fit(X_train, y_train)
final_pred = final_pipeline.predict(X_test)
final_mae = mean_absolute_error(y_test, final_pred)
final_r2 = r2_score(y_test, final_pred)
print(f"Final pipeline MAE: {final_mae:.2f}, R²: {final_r2:.3f}")

# ==============================
# Tests pytest
# ==============================
import pytest

def test_columns():
    required_columns = selected_features + [target_col]
    missing = [col for col in required_columns if col not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"

def test_model_performance():
    pred = final_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    assert mae < 15, f"MAE too high: {mae}"
