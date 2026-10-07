from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

NORMAL = {
    "duration": 0, "protocol_type": "tcp", "service": "ftp_data", "flag": "SF",
    "src_bytes": 491, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
    "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
    "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
    "is_guest_login": 0, "count": 2, "srv_count": 2, "serror_rate": 0,
    "srv_serror_rate": 0, "rerror_rate": 0, "srv_rerror_rate": 0, "same_srv_rate": 1,
    "diff_srv_rate": 0, "srv_diff_host_rate": 0, "dst_host_count": 150,
    "dst_host_srv_count": 25, "dst_host_same_srv_rate": 0.17, "dst_host_diff_srv_rate": 0.03,
    "dst_host_same_src_port_rate": 0.17, "dst_host_srv_diff_host_rate": 0,
    "dst_host_serror_rate": 0, "dst_host_srv_serror_rate": 0, "dst_host_rerror_rate": 0.05,
    "dst_host_srv_rerror_rate": 0,
}

ATTACK = {
    **NORMAL,
    "service": "private", "flag": "S0", "src_bytes": 0, "count": 123,
    "srv_count": 6, "serror_rate": 1, "srv_serror_rate": 1, "same_srv_rate": 0.05,
    "diff_srv_rate": 0.07, "dst_host_count": 255, "dst_host_srv_count": 26,
    "dst_host_same_srv_rate": 0.10, "dst_host_diff_srv_rate": 0.05,
    "dst_host_same_src_port_rate": 0, "dst_host_serror_rate": 1,
    "dst_host_srv_serror_rate": 1, "dst_host_rerror_rate": 0,
    "dst_host_srv_rerror_rate": 0,
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["features"] == 41

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model"]
    assert "feature_importance" in response.json()

def test_normal_prediction():
    response = client.post("/predict", json=NORMAL)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Normal", "Attack"}
    assert body["features_received"] == 41
    assert 0 <= body["confidence"] <= 1

def test_attack_prediction():
    response = client.post("/predict", json=ATTACK)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "Attack"
    assert 0 <= body["confidence"] <= 1


def test_predict_batch_endpoint():
    normal = {
        "duration": 0, "protocol_type": "tcp", "service": "ftp_data", "flag": "SF",
        "src_bytes": 491, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
        "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0, "root_shell": 0,
        "su_attempted": 0, "num_root": 0, "num_file_creations": 0, "num_shells": 0,
        "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 2, "srv_count": 2, "serror_rate": 0, "srv_serror_rate": 0, "rerror_rate": 0,
        "srv_rerror_rate": 0, "same_srv_rate": 1, "diff_srv_rate": 0, "srv_diff_host_rate": 0,
        "dst_host_count": 150, "dst_host_srv_count": 25, "dst_host_same_srv_rate": 0.17,
        "dst_host_diff_srv_rate": 0.03, "dst_host_same_src_port_rate": 0.17,
        "dst_host_srv_diff_host_rate": 0, "dst_host_serror_rate": 0, "dst_host_srv_serror_rate": 0,
        "dst_host_rerror_rate": 0.05, "dst_host_srv_rerror_rate": 0,
    }
    response = client.post("/predict-batch", json=[normal, normal])
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["features_received"] == 41
