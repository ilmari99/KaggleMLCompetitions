# Visualize the Titanic dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

def show_data_as_text(data):
    print(data.head())
    print(data.describe())
    
def show_distrbutions(data):
    data.hist(bins=50, figsize=(20,15))
    plt.show()
    
def convert_to_onehot(data, columns=[None]):
    """ Convert columns to one-hot-encoded columns.
    So values from 1-5 will be converted to 5 columns with 0s and 1s.
    """
    for col in columns:
        data[col] = data[col].astype(str)
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col,dtype=int)], axis=1)
        data.drop(col, axis=1, inplace=True)
    return data
    
def load_data(path, convert_cabin=True, drop_non_num_cols=True, convert_categories_to_onehot=True, n_most_info_cols=10, only_take_cols=None):
    data = pd.read_csv(path)
    data["Sex"] = data["Sex"].map({"male":1, "female":0})
    data["Embarked"] = data["Embarked"].map({"S":0, "C":1, "Q":2})
    # Convert cabins to their first letter column (mapped to 0, 1, 2, 3, 4, 5, 6, 7, 8, 9) and to cabin number column
    if convert_cabin:
        # Make sure all cabins are of the form "X###" where X is a letter and ### is a number
        # If not (or cabin is NaN), set the cabin to "U0" (unknown cabin)
        data["Cabin"] = data["Cabin"].map(lambda x: "U0" if pd.isnull(x) else x)
        data["Cabin"] = data["Cabin"].map(lambda x: "U0" if not str.isalpha(x[0]) else x)
        data["Cabin"] = data["Cabin"].map(lambda x: "U0" if not str.isnumeric(x[1:].split(" ")[0]) else x)
        # Take the first letter of the cabin and convert it to a number for CabinLetter column
        data["CabinLetter"] = data["Cabin"].map(lambda x: x[0])
        data["CabinLetter"] = data["CabinLetter"].map({"U":0, "C":1, "E":2, "G":3, "D":4, "A":5, "B":6, "F":7, "T":8})
        # Take the number of the cabin (rest of the Cabin value) and convert it to a number for CabinNumber column
        data["CabinNumber"] = data["Cabin"].map(lambda x: x[1:].split(" ")[0])
        data["CabinNumber"] = data["CabinNumber"].astype(int)
    data.drop("Cabin", axis=1, inplace=True)
    if drop_non_num_cols:
        data.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    if "Survived" in data.columns:
        data.dropna(subset=["Survived"], inplace=True)
    if convert_categories_to_onehot:
        # Convert Pclass, sibsp, parch, embarked, and cabinLetter to one-hot-encoded columns
        data = convert_to_onehot(data, columns=["Pclass", "SibSp", "Parch", "Embarked"])
    if only_take_cols is None:
        infos = measure_mutual_information(data, target_col="Survived")
        # Take the n_most_info_cols most informative columns
        n_most_info_cols = len(infos) - 1 if n_most_info_cols == "max" else n_most_info_cols
        most_info_cols = ["Survived"] + infos[:n_most_info_cols].index.tolist() 
        data = data[most_info_cols]
    else:
        print(data.columns)
        data = data[only_take_cols]
    return data

def measure_mutual_information(data, target_col="Survived", show = False):
    """ Measure the mutual information between all columns and the target column.
    """
    data = data.dropna()
    y = data.pop(target_col)
    X = data
    # Calculate mutual information between each feature and the target
    discrete_mask_array = []
    for idx, col in enumerate(X.columns):
        if col in ["Age", "Fare", "CabinNumber", "CabinLetter"]:
            discrete_mask_array.append(False)
        else:
            discrete_mask_array.append(True)
    mi = mutual_info_classif(X, y, discrete_features=discrete_mask_array)
    mi = pd.Series(mi)
    mi.index = X.columns
    mi.sort_values(ascending=False, inplace=True)
    if show:
        mi.plot.bar(figsize=(20, 8))
        plt.show()
    return mi

    
    
    
if __name__ == "__main__":
    df = load_data("Titanic/titanic/train.csv", convert_cabin=True, drop_non_num_cols=True, convert_categories_to_onehot = True, n_most_info_cols=10)
    measure_mutual_information(df,show=True)
    
    # Count nans
    print(df.isnull().sum())
    
    show_data_as_text(df)
    show_distrbutions(df)
    
    