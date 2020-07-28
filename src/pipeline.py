# Grid and Random Search with Pipeline
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing  # for scaling the data
from sklearn import decomposition
from sklearn import pipeline

if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis = 1).values #Features
    y = df.price_range.values

    sca = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)

    classifier = pipeline.Pipeline(
        [
            ("scaling", sca),
            ("pca", pca),
            ("rf", rf)
        ]
    )

    param_grid = {
        "pca__n_components": np.arange(5, 10),
        "rf__n_estimators": np.arange(100, 1500, 100),
        "rf__max_depth": np.arange(1, 20),
        "rf__criterion": ["gini", "entropy"],
    }
    # Random search is not as expensive as grid search
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        n_jobs=1,
        cv=5,
    )

    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())