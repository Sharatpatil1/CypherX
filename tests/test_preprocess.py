from pathlib import Path
import json
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "ml" / "models"

def test_artifacts_exist():
    for name in ["intrusion_model.joblib", "preprocessor.joblib", "feature_columns.json", "metadata.json"]:
        assert (MODEL / name).exists(), name

def test_feature_schema():
    info = json.loads((MODEL / "feature_columns.json").read_text(encoding="utf-8"))
    assert len(info["raw_features"]) == 41
    assert info["categorical_features"] == ["protocol_type", "service", "flag"]
    assert len(info["numerical_features"]) == 38

def test_artifacts_load():
    assert joblib.load(MODEL / "intrusion_model.joblib") is not None
    assert joblib.load(MODEL / "preprocessor.joblib") is not None
