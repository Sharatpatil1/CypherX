from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"
FEATURES = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
COLS = FEATURES + ["label","difficulty"]
CAT = ["protocol_type","service","flag"]
NUM = [x for x in FEATURES if x not in CAT]
DOS={"back","land","neptune","pod","smurf","teardrop","apache2","udpstorm","processtable","mailbomb"}
PROBE={"ipsweep","nmap","portsweep","satan","mscan","saint"}
R2L={"ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster","sendmail","named","snmpgetattack","snmpguess","xlock","xsnoop","httptunnel"}
U2R={"buffer_overflow","loadmodule","perl","rootkit","ps","sqlattack","xterm"}
def category(x):
    x=str(x).strip().rstrip(".")
    if x=="normal": return "Normal"
    if x in DOS: return "DoS"
    if x in PROBE: return "Probe"
    if x in R2L: return "R2L"
    if x in U2R: return "U2R"
    return "Other"
def load(path):
    return pd.read_csv(path,header=None,names=COLS)
def main():
    MODELS.mkdir(parents=True,exist_ok=True)
    tr,te=load(DATA/"KDDTrain+.txt"),load(DATA/"KDDTest+.txt")
    for df in (tr,te):
        df["label"]=df["label"].astype(str).str.strip().str.rstrip(".")
        df["target"]=(df["label"]!="normal").astype(int)
        df["attack_category"]=df["label"].map(category)
    pp=ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CAT)])
    Xt=pp.fit_transform(tr[FEATURES]); Xv=pp.transform(te[FEATURES])
    joblib.dump(pp,MODELS/"preprocessor.joblib")
    joblib.dump({"X":Xt,"y":tr.target.to_numpy(),"labels":tr.label.to_numpy(),"attack_categories":tr.attack_category.to_numpy()},MODELS/"processed_train.joblib")
    joblib.dump({"X":Xv,"y":te.target.to_numpy(),"labels":te.label.to_numpy(),"attack_categories":te.attack_category.to_numpy()},MODELS/"processed_test.joblib")
    json.dump({"raw_features":FEATURES,"categorical_features":CAT,"numerical_features":NUM,"processed_feature_count":int(Xt.shape[1])},open(MODELS/"feature_columns.json","w"),indent=2)
    print(f"Training samples: {len(tr):,}; Testing samples: {len(te):,}; Processed features: {Xt.shape[1]}")
if __name__=="__main__": main()
