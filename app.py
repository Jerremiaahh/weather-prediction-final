from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
FEATURES = ["Temperature", "Humidity", "WindSpeed", "Rainfall"]
TARGET = "Weather"

def demo_dataset():
    rows = [
        [30,70,12,0,"Clear"],[32,65,10,0,"Clear"],[28,75,8,1,"Partly Cloudy"],
        [27,82,14,6,"Rain"],[25,90,18,15,"Rain"],[29,78,12,4,"Partly Cloudy"],
        [34,55,9,0,"Clear"],[31,68,11,0,"Clear"],[26,88,20,18,"Rain"],
        [24,92,16,15,"Rain"],[29,80,10,3,"Partly Cloudy"],[33,60,7,0,"Clear"],
        [28,76,13,2,"Partly Cloudy"],[27,85,17,9,"Rain"],[35,52,8,0,"Clear"],
        [30,73,9,1,"Partly Cloudy"],[26,87,19,13,"Rain"],[32,62,10,0,"Clear"],
        [29,79,15,5,"Partly Cloudy"],[25,91,21,20,"Rain"],[34,58,6,0,"Clear"],
        [28,81,11,4,"Partly Cloudy"],[27,89,18,12,"Rain"],[31,66,8,0,"Clear"],
        [30,77,13,3,"Partly Cloudy"],[24,94,22,25,"Rain"],[33,57,9,0,"Clear"],
        [29,74,10,2,"Partly Cloudy"],[26,86,16,10,"Rain"],[35,50,7,0,"Clear"],
        [31,70,12,1,"Partly Cloudy"],[25,93,20,19,"Rain"],[32,63,8,0,"Clear"]
    ]
    return pd.DataFrame(rows, columns=FEATURES+[TARGET])

def load_data():
    path = "weatheraus.csv"
    if os.path.exists(path) and os.path.getsize(path) > 30:
        try:
            df = pd.read_csv(path)
            rename = {}
            aliases = {
                "temperature":"Temperature","temp":"Temperature",
                "humidity":"Humidity","windspeed":"WindSpeed","wind":"WindSpeed",
                "rainfall":"Rainfall","rain":"Rainfall","weather":"Weather",
                "condition":"Weather","raintomorrow":"Weather"
            }
            for c in df.columns:
                k = c.strip().lower().replace("_","").replace(" ","")
                if k in aliases: rename[c] = aliases[k]
            df = df.rename(columns=rename)
            if "Weather" not in df.columns and "RainTomorrow" in df.columns:
                df["Weather"] = df["RainTomorrow"].map(
                    lambda x: "Rain" if str(x).lower().strip() in ["yes","y","1","rain"] else "Clear"
                )
            if all(c in df.columns for c in FEATURES+[TARGET]):
                out = df[FEATURES+[TARGET]].copy()
                for c in FEATURES: out[c] = pd.to_numeric(out[c], errors="coerce")
                out[TARGET] = out[TARGET].astype(str).str.strip()
                out = out.dropna()
                if len(out) >= 20 and out[TARGET].nunique() >= 2:
                    return out, "Project dataset"
        except Exception:
            pass
    return demo_dataset(), "Demo dataset"

def train():
    df, source = load_data()
    X, y = df[FEATURES], df[TARGET]
    try:
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(Xtr,ytr)
    pred = model.predict(Xte)
    m = {
        "accuracy":round(accuracy_score(yte,pred)*100,1),
        "precision":round(precision_score(yte,pred,average="weighted",zero_division=0)*100,1),
        "recall":round(recall_score(yte,pred,average="weighted",zero_division=0)*100,1),
        "f1":round(f1_score(yte,pred,average="weighted",zero_division=0)*100,1),
        "samples":len(df), "source":source
    }
    return model,m

MODEL,METRICS=train()

def icon(c):
    c=c.lower()
    if "rain" in c: return "🌧️"
    if "cloud" in c: return "⛅"
    if "clear" in c: return "☀️"
    return "🌤️"

@app.route("/")
def home(): return render_template("index.html",metrics=METRICS)

@app.route("/predict",methods=["POST"])
def predict():
    try:
        d=request.get_json()
        vals=[float(d["temperature"]),float(d["humidity"]),float(d["wind"]),float(d["rainfall"])]
        row=pd.DataFrame([vals],columns=FEATURES)
        result=str(MODEL.predict(row)[0])
        confidence=round(float(np.max(MODEL.predict_proba(row)))*100,1)
        return jsonify({"prediction":result,"icon":icon(result),"confidence":confidence})
    except Exception as e:
        return jsonify({"error":str(e)}),400

if __name__=="__main__":
    app.run(debug=True)
