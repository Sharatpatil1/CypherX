from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent.parent
M=BASE/"models"; R=BASE/"results"
def main():
    model=joblib.load(M/"intrusion_model.joblib")
    pp=joblib.load(M/"preprocessor.joblib")
    names=list(pp.named_transformers_["num"].get_feature_names_out())+list(pp.named_transformers_["cat"].get_feature_names_out())
    values=model.feature_importances_ if hasattr(model,"feature_importances_") else abs(model.coef_[0])
    df=pd.DataFrame({"feature":names,"importance":values}).sort_values("importance",ascending=False)
    df.to_csv(R/"feature_importance.csv",index=False)
    top=df.head(15).sort_values("importance")
    plt.figure(figsize=(9,6)); plt.barh(top.feature,top.importance); plt.xlabel("Importance"); plt.title("CypherX Feature Importance"); plt.tight_layout(); plt.savefig(R/"feature_importance.png",dpi=160); plt.close()
    print(df.head(15).to_string(index=False))
if __name__=="__main__": main()
