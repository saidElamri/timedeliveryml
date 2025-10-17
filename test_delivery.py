from EDA import final_pipeline, df, selected_features, target_col, X_test, y_test
import pytest
from sklearn.metrics import mean_absolute_error

def test_columns():
    required_columns = selected_features + [target_col]
    missing = [col for col in required_columns if col not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"

def test_model_performance():
    pred = final_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    assert mae < 15, f"MAE too high: {mae}"
