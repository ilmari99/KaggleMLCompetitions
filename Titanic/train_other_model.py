# Train a random forest binary classifier on the Titanic dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# SVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# Random grid search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, f1_score
from visualize import load_data
from sklearn.preprocessing import MinMaxScaler
    
if __name__ == "__main__":
    # sst seeds
    np.random.seed(42)
    df_train = load_data("Titanic/titanic/train.csv",n_most_info_cols=10)
    df_train.rename({'Embarked_1.0': 'Embarked_1', 'Embarked_0.0': 'Embarked_0','Embarked_2.0' : 'Embarked_2'}, axis=1, inplace=True, errors="ignore")
    # Cnvert nans to -1
    df_train.fillna(-1, inplace=True)
    # Split into X and y
    y = df_train.pop("Survived")
    X = df_train
    # Split to train and validation
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True, random_state=42)
    X_train = X
    y_train = y
    
    # Do a random grid search to find the best hyperparameters
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
    
    # Minmax scale the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    
    # Create the base model to tune
    estim = RandomForestClassifier()
    param_space = random_forest_param_space
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    # Best model is the model that maximizes accuracy
    estim_random = RandomizedSearchCV(estimator=estim, param_distributions=param_space, n_iter=100, verbose=2, random_state=42, n_jobs=-1, scoring="accuracy")
    # Fit the random search model
    estim_random.fit(X_train, y_train)
    print("Best params:", estim_random.best_params_)
    print("Best score:", estim_random.best_score_)
    model = estim_random.best_estimator_   
    
    # Only take the Survived column, which is the first column
    #preds = model.predict(X_test)
    #report = classification_report(y_test, preds)
    #accuracy = accuracy_score(y_test, preds)
    #kappa = cohen_kappa_score(y_test, preds)
    #macro_f1 = f1_score(y_test, preds, average="macro")
    #percent_survived = np.sum(y_test) / len(y_test)
    #print("Percent survived:", percent_survived)
    #print("Total accuracy:", accuracy)
    #print("Kappa:", kappa)
    #print("Classification report:")
    #print(report)
    
    # Create submission
    select_columns = df_train.columns.values.tolist() + ["PassengerId"]
    print(select_columns)
    df_test = load_data("Titanic/titanic/test.csv",drop_non_num_cols=False, only_take_cols=select_columns) 
    pass_id = df_test["PassengerId"]
    df_test.drop(["PassengerId"], axis=1, inplace=True)
    # Order the columns so they are in the same order as the training data
    df_test = df_test[df_train.columns]
    # Convert nans to -1
    df_test.fillna(-1, inplace=True) 
    
    scaler = MinMaxScaler()
    df_test = scaler.fit_transform(df_test)
    # Predictions
    y_pred = model.predict(df_test)
    # Output predictions
    df_out = pd.DataFrame({"PassengerId": pass_id, "Survived": y_pred})
    df_out.to_csv("Titanic/titanic/submission.csv", index=False)
    
    
    
