""" Train a model, that takes in a horse medical vector, with masked values, and outputs the same vector, with the masked values filled in.
"""
#Disable FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import pandas as pd
import numpy as np
from load_data import modify_data, convert_back_to_int
from sklearn.model_selection import train_test_split
import tensorflow as tf
# tf addons
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
# SVC
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

if __name__ == "__main__":
    df = pd.read_csv("HorseMod/train.csv")
    # Add horse.csv data. In this file, all values are in double quotes
    horse_df = pd.read_csv("HorseMod/horse.csv")
    df = pd.concat([df, horse_df], axis=0)
    print(df.head())
    df = df.sample(frac=1)
    df = modify_data(df,get_columns=40)
    df.rename(columns={"outcome_outcome_died": "outcome_died", "outcome_outcome_euthanized": "outcome_euthanized", "outcome_outcome_lived": "outcome_lived"}, inplace=True)
    df = convert_back_to_int(df, ["outcome_died", "outcome_euthanized", "outcome_lived"], new_col_name="outcome")
    y = df.pop("outcome")
    X = df
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    random_forest_param_space = {
        "n_estimators": [int(x) for x in np.linspace(start=10, stop=1000, num=10)],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_features": [None, "sqrt", "log2"],
        "max_depth": [None] + [int(x) for x in np.linspace(1, 110, num=11)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    
    SVC_param_space = {
        "C": [0.1, 1, 10, 100, 500],
        "kernel": ["rbf"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced", None],
        "break_ties": [True, False],
        "tol": [1e-3, 1e-4, 1e-5],
        "shrinking": [True, False],
    }
    
    gradient_boosting_param_space = {
        "loss": ["log_loss", "exponential"],
        "learning_rate": [0.2, 0.1, 0.05, 0.01],
        "n_estimators": [int(x) for x in np.linspace(start=10, stop=1000, num=10)],
        "subsample": [0.5, 0.75, 1.0],
        "criterion": ["friedman_mse", "squared_error"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    
    estimator = RandomForestClassifier()
    param_space = random_forest_param_space
    
    # F1 scorer with micor average wrapper
    def scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        return f1_score(y, y_pred, average="micro")
    estim_random = RandomizedSearchCV(estimator=estimator, param_distributions=param_space, cv=4, n_iter=160, verbose=2, random_state=42, n_jobs=-1, scoring=scorer, refit=True)
    # Fit the random search model
    estim_random.fit(X_train, y_train)
    print("Best params:", estim_random.best_params_)
    print("Best score:", estim_random.best_score_)
    model = estim_random.best_estimator_
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average="micro")
    
    print(report)
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1_micro}")
    
    
    
    df_test = pd.read_csv("HorseMod/test.csv")
    horse_ids = df_test["id"]
    train_cols = df.columns.values.tolist()
    print(train_cols)
    df_test = modify_data(df_test,get_columns=train_cols)
    
    # Make sure the columns are in the same order as in df
    X_test : pd.DataFrame = df_test[train_cols]
    X_test = scaler.transform(X_test)
    
    test_preds = model.predict(X_test) 
    
    
    # Convert preds to {0: died, 1: euthanized, 2: lived}
    preds = pd.DataFrame({"id": horse_ids, "outcome": test_preds})
    preds["outcome"].replace({0: "died", 1: "euthanized", 2: "lived"}, inplace=True)
    df_submission = pd.DataFrame({"id": horse_ids, "outcome": preds["outcome"]})
    df_submission.to_csv("HorseMod/submission.csv", index=False)
    
    
    
    
    
    
    






