""" Load data to a pandas dataframe, and preprocess the variables. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

def modify_leasion_data(df):
    """ Modify data about lesions.
    Each number is 1-4 digits long. If it is 5 digits OR just the number 0, it is replaced with -1.
    - first number is site of lesion 
    1 = gastric 
    2 = sm intestine 
    3 = lg colon 
    4 = lg colon and cecum 
    5 = cecum 
    6 = transverse colon 
    7 = retum/descending colon 
    8 = uterus 
    9 = bladder 
    11 = all intestinal sites 
    00 = none 
    - second number is type 
    1 = simple 
    2 = strangulation 
    3 = inflammation 
    4 = other 
    - third number is subtype 
    1 = mechanical 
    2 = paralytic 
    0 = n/a 
    - fourth number is specific code 
    1 = obturation 
    2 = intrinsic 
    3 = extrinsic 
    4 = adynamic 
    5 = volvulus/torsion 
    6 = intussuption 
    7 = thromboembolic 
    8 = hernia 
    9 = lipoma/slenic incarceration 
    10 = displacement 
    0 = n/a

    These codes are converted to 4 columns:
    - lesion_1 -> lesion_11, lesion_12, lesion_13, lesion_14
    - lesion_2 -> lesion_21, lesion_22, lesion_23, lesion_24
    - lesion_3 -> lesion_31, lesion_32, lesion_33, lesion_34
    """
    lesion_columns = ["lesion_1","lesion_2","lesion_3"]
    for column in lesion_columns:
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(lambda x: "0" if x == "-1" else x)
        df[column] = df[column].apply(lambda x: "0" if len(x) > 5 else x)
        df[column + "1"] = df[column].apply(lambda x: x[0] if len(x) > 0 else "-1").astype(int)
        df[column + "2"] = df[column].apply(lambda x: x[1] if len(x) > 1 else "-1").astype(int)
        df[column + "3"] = df[column].apply(lambda x: x[2] if len(x) > 2 else "-1").astype(int)
        df[column + "4"] = df[column].apply(lambda x: x[3] if len(x) > 3 else "-1").astype(int)
        df.drop(column, axis=1, inplace=True)
    # Move 'outcome' column to the end if it exists
    if "outcome" in df.columns:
        df = df[[c for c in df if c not in ['outcome']] + ['outcome']]
    return df

def modify_data(df, features_to_onehot="default", get_columns="all"):
    """ 
    Find unique values for each variable and replace them with integers.
    """
    features_to_onehot_default = ["outcome", "surgery", "age", "temp_of_extremities","peripheral_pulse", "mucous_membrane","capillary_refill_time", "pain","peristalsis","abdominal_distention","nasogastric_tube","nasogastric_reflux",
                          "rectal_exam_feces","abdomen", "abdomo_appearance", "surgical_lesion", "cp_data"]
    df = df.fillna(-1)
    # Modify lesion data
    df = modify_leasion_data(df)
    df.drop(["id","hospital_number"], axis=1, inplace=True)
    # Select either all object types, or a list of features
    features = features_to_onehot_default if features_to_onehot == "default" else features_to_onehot
    if "outcome" not in df.columns and "outcome" in features:
        features.remove("outcome")
    df = convert_to_onehot(df, columns = features)
    if "outcome" in df.columns:
        # Move 'outcome' columns to the front
        df = df[[c for c in df if "outcome" in c] + [c for c in df if "outcome" not in c]]
    
    if isinstance(get_columns, list):
        print(df.columns)
        df = df[get_columns]
    elif isinstance(get_columns, int):
        df = convert_back_to_int(df, ["outcome_died", "outcome_euthanized", "outcome_lived"], new_col_name="outcome")
        mi = measure_mutual_information(df, target_col="outcome")
        # Select the top n columns
        top_cols = ["outcome"] + list(mi.index[:get_columns])
        df = df[top_cols]
        df = convert_to_onehot(df, columns = ["outcome"])
        # Move 'outcome' columns to the front
        df = df[[c for c in df if "outcome" in c] + [c for c in df if "outcome" not in c]]
    return df

def convert_to_onehot(data, columns=[None]):
    """ Convert columns to one-hot-encoded columns.
    So values from 1-5 will be converted to 5 columns with 0s and 1s.
    """
    for col in columns:
        if len(data[col].unique()) <= 2:
            #Map the unique values to 0 and 1
            uniques = data[col].unique()
            data[col] = data[col].map({uniques[0]: 0, uniques[1]: 1})
            continue
        data[col] = data[col].astype(str)
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col,dtype=int)], axis=1)
        data.drop(col, axis=1, inplace=True)
    return data

def convert_back_to_int(data, columns, new_col_name = None):
    """ Convert one-hot-encoded columns back to a single column.
    for example [[0,0,1], [1, 0, 0]] -> [2, 0]
    """
    new_col_name = columns[0].split("_")[0] if new_col_name is None else new_col_name
    data[new_col_name] = data[columns].idxmax(axis=1)
    data.drop(columns, axis=1, inplace=True)
    return data

def measure_mutual_information(data, target_col="outcome", show = False):
    """ Measure the mutual information between all columns and the target column.
    """
    data = data.dropna()
    y = data.pop(target_col)
    X = data
    mi = mutual_info_classif(X, y)
    mi = pd.Series(mi)
    mi.index = X.columns
    mi.sort_values(ascending=False, inplace=True)
    if show:
        mi.plot.bar(figsize=(20, 8))
        plt.show()
    return mi
    

def plot_histograms(data):
    data.hist(bins=50, figsize=(20,15))
    plt.show()

def check_data(df):
    """ Check data for missing values. """
    print(df.isnull().sum())
    print(df.head())
    print(df.describe())
    print(df.dtypes)


if __name__ == "__main__":
    df = pd.read_csv("HorseMod/train.csv")
    check_data(df)
    # Modify data
    df = modify_data(df)
    df = convert_back_to_int(df, ["outcome_died", "outcome_euthanized", "outcome_lived"])
    measure_mutual_information(df, show=True)
    df = convert_to_onehot(df, columns=["outcome"])
    check_data(df)
    plot_histograms(df)