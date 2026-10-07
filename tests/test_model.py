import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_metadata_metrics():
    metadata = json.loads((ROOT / "ml/models/metadata.json").read_text(encoding="utf-8"))
    metrics = metadata["metrics"]
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert 0 <= metrics[key] <= 1
    assert metadata["model"] == "Decision Tree"
