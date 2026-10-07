from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

BASE=Path(__file__).resolve().parent.parent
M=BASE/"models"; R=BASE/"results"
def metric(y,p):
    return {"accuracy":accuracy_score(y,p),"precision":precision_score(y,p,zero_division=0),"recall":recall_score(y,p,zero_division=0),"f1":f1_score(y,p,zero_division=0)}
def main():
    R.mkdir(exist_ok=True)
    tr=joblib.load(M/"processed_train.joblib"); te=joblib.load(M/"processed_test.joblib")
    X,y=tr["X"],tr["y"]; Xt,yt=te["X"],te["y"]
    models={
      "Logistic Regression":LogisticRegression(max_iter=1000,n_jobs=None),
      "Decision Tree":DecisionTreeClassifier(max_depth=28,random_state=42,class_weight="balanced"),
      "Random Forest":RandomForestClassifier(n_estimators=150,max_depth=28,n_jobs=-1,random_state=42,class_weight="balanced_subsample")
    }
    results=[]; fitted={}
    for name,model in models.items():
        print("Training",name)
        model.fit(X,y); p=model.predict(Xt); m=metric(yt,p); m["model"]=name; results.append(m); fitted[name]=model
    best=max(results,key=lambda x:x["f1"])["model"]; model=fitted[best]
    p=model.predict(Xt); cm=confusion_matrix(yt,p).tolist()
    joblib.dump(model,M/"intrusion_model.joblib")
    pd.DataFrame(results).to_csv(R/"model_comparison.csv",index=False)
    pd.DataFrame(classification_report(yt,p,output_dict=True,zero_division=0)).T.to_csv(R/"classification_report.csv")
    # attack-category performance from true NSL-KDD labels on the test split
    cats=pd.Series(te["attack_categories"])
    rows=[]
    for c in ["Normal","DoS","Probe","R2L","U2R"]:
        mask=cats.to_numpy()==c
        if mask.sum():
            rows.append({"category":c,"samples":int(mask.sum()),"attack_detection_recall":float((p[mask]==1).mean()) if c!="Normal" else float((p[mask]==0).mean())})
    pd.DataFrame(rows).to_csv(R/"attack_category_analysis.csv",index=False)
    # Feature importances mapped back through one-hot names.
    pp=joblib.load(M/"preprocessor.joblib")
    names=list(pp.named_transformers_["num"].get_feature_names_out())+list(pp.named_transformers_["cat"].get_feature_names_out())
    if hasattr(model,"feature_importances_"):
        vals=model.feature_importances_
    else:
        vals=np.abs(model.coef_[0])
    fi=pd.DataFrame({"feature":names,"importance":vals}).sort_values("importance",ascending=False)
    fi.to_csv(R/"feature_importance.csv",index=False)
    try:
        import matplotlib.pyplot as plt
        top=fi.head(15).sort_values("importance")
        plt.figure(figsize=(9,6)); plt.barh(top.feature,top.importance); plt.xlabel("Importance"); plt.title(f"CypherX — {best} Feature Importance"); plt.tight_layout(); plt.savefig(R/"feature_importance.png",dpi=160); plt.close()
        plt.figure(figsize=(5,4)); plt.imshow(cm,aspect="auto"); plt.xticks([0,1],["Normal","Attack"]); plt.yticks([0,1],["Normal","Attack"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("CypherX Confusion Matrix"); 
        for i in range(2):
            for j in range(2): plt.text(j,i,str(cm[i][j]),ha="center",va="center")
        plt.tight_layout(); plt.savefig(R/"confusion_matrix.png",dpi=160); plt.close()
    except Exception as e: print("Plot generation warning:",e)
    metadata={"model":best,"metrics":metric(yt,p),"confusion_matrix":cm,"model_comparison":results,"test_distribution":cats.value_counts().to_dict(),"attack_category_analysis":rows,"feature_importance":fi.head(15).to_dict(orient="records")}
    json.dump(metadata,open(M/"metadata.json","w"),indent=2)
    print(json.dumps(metadata["metrics"],indent=2)); print("Best model:",best)
if __name__=="__main__": main()
