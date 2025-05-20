from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("data/parkinsons.data").drop(['name'], axis=1)
X = df.drop("status", axis=1)
y = df["status"]

params = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring="accuracy", cv=5)
grid.fit(X, y)
print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)
