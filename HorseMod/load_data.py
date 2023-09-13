""" Load data to a pandas dataframe, and preprocess the variables. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    # Move 'outcome' column to the end
    df = df[[c for c in df if c not in ['outcome']] + ['outcome']]
    return df

def modify_data(df, fill_missing=True, features="all"):
    """ 
    Find unique values for each variable and replace them with integers.
    """
    if fill_missing:
        # Fill missing values with -1
        df = df.fillna(-1)
    # Modify lesion data
    df = modify_leasion_data(df)
    df.drop(["id","hospital_number"], axis=1, inplace=True)
    # Select either all object types, or a list of features
    features = df.select_dtypes(include="object").columns if features == "all" else features
    for feature in features:
        # Find unique values
        unique_values = df[feature].unique()
        # Replace values with integers
        df[feature] = df[feature].replace(unique_values, np.arange(len(unique_values)))
    return df



def plot_histograms(df,nrows=5):
    """ Plot histograms for each variable. """
    numeric_cols = df.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots(nrows=nrows, ncols=nrows, figsize=(10, 10))
    for i, feature in enumerate(numeric_cols):
        ax[i//nrows, i%nrows].hist(df[feature])
        ax[i//nrows, i%nrows].set_title(feature)
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
    plot_histograms(df)
    # Modify data
    df = modify_data(df)
    check_data(df)
    plot_histograms(df,nrows=6)