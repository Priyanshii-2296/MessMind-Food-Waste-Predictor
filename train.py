import pandas as pd, numpy as np, joblib, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/mess_data.csv")
X = df.drop(columns=["date", "food_waste_kg"]); y = df["food_waste_kg"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
          "Gradient Boosting": GradientBoostingRegressor(random_state=42)}
res = {}
for n, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    res[n] = {"MAE": round(mean_absolute_error(yte, p), 2), "R2": round(r2_score(yte, p), 3)}
best = min(res, key=lambda k: res[k]["MAE"]); bm = models[best]
joblib.dump(bm, "models/best_model.joblib")
json.dump({"results": res, "best": best, "features": list(X.columns)}, open("models/metrics.json", "w"), indent=2)
print(json.dumps(res, indent=1), "\nBest:", best)

plt.figure(figsize=(6, 3.6)); plt.bar(res.keys(), [v["MAE"] for v in res.values()], color=["#9aa5b1", "#2f855a", "#2b6cb0"])
plt.ylabel("MAE (kg of food waste)"); plt.title("Model comparison (lower is better)"); plt.tight_layout(); plt.savefig("charts/model_comparison.png", dpi=150); plt.close()
plt.figure(figsize=(4.6, 4.6)); plt.scatter(yte, bm.predict(Xte), s=10, alpha=.6, color="#2f855a")
lim = [yte.min(), yte.max()]; plt.plot(lim, lim, "r--"); plt.xlabel("Actual waste (kg)"); plt.ylabel("Predicted waste (kg)")
plt.title(f"{best}: actual vs predicted"); plt.tight_layout(); plt.savefig("charts/actual_vs_predicted.png", dpi=150); plt.close()
if hasattr(bm, "feature_importances_"):
    imp = pd.Series(bm.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(6, 3.6)); imp.plot.barh(color="#2f855a"); plt.title("What drives food waste?"); plt.tight_layout()
    plt.savefig("charts/feature_importance.png", dpi=150); plt.close()
