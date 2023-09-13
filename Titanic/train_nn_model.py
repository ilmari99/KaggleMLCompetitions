# Train a random forest binary classifier on the Titanic dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# Random grid search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, f1_score
from visualize import load_data
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class CustomLoss(tf.keras.losses.Loss):
    """ Loss function that masks the output and only calculates loss for non-masked values.
    """
    def __init__(self):
        super(CustomLoss, self).__init__()
        
    def call(self, y_true, y_pred):
        # y_true and y_pred are vectors of shape (batch_size, 35)
        survived_loss = tf.keras.losses.binary_crossentropy(y_true[:, 0], tf.sigmoid(y_pred[:, 0]))
        y_true = y_true[:, 1:]
        y_pred = y_pred[:, 1:]
        mask = tf.math.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        mse_vector_loss = tf.keras.losses.MSE(y_true, y_pred) * 0.001
        return survived_loss + mse_vector_loss

def get_model(input_shape, output_shape):
    """ Create a model, that takes in a vector of input_shape with mask and outputs a vector of output_shape,
    which is the original input vector without mask.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer="l2")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer="l2")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer="l2")(x)
    outputs = tf.keras.layers.Dense(output_shape, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=CustomLoss())
    return model
    

if __name__ == "__main__":
    # sst seeds
    np.random.seed(42)
    df_train = load_data("Titanic/titanic/train.csv",n_most_info_cols=30)
    df_train.pop(f"Embarked_nan")
    df_train.rename({'Embarked_1.0': 'Embarked_1', 'Embarked_0.0': 'Embarked_0','Embarked_2.0' : 'Embarked_2'}, axis=1, inplace=True, errors="ignore")
    df_train = df_train.sample(frac=1, random_state=42)
    #df_train.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    # Drop rows where Survived = NaN
    df_train.dropna(subset=["Survived"], inplace=True)
    # Cnvert nans to -1
    df_train.fillna(-1, inplace=True)
    # Split into X and y
    y = df_train
    X = df_train
    # Split to train and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=y.columns)
    y_test = pd.DataFrame(y_test, columns=y.columns)
    
    # Randomly mask 20% of X_train values
    mask = np.random.choice([True, False], size=X_train.shape, p=[0.2, 0.8])
    X_train[mask] = -1
    
    # Convert 50% of 'Survived' to missing (-1)
    X_train["Survived"] = -1#np.where(np.random.choice([True, False], size=X_train.shape[0], p=[0.5, 0.5]), -1, X_train["Survived"])
    # Convert all X_test values to -1
    X_test["Survived"] = -1
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # Random forest classifier
    model = get_model(X_train.shape[1], y_train.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)
    model.fit(X_train, y_train, epochs=5000, validation_data=(X_test, y_test), callbacks=[early_stopping, tensorboard], batch_size=128)
    
    # Only take the Survived column, which is the first column
    preds = model.predict(X_test)
    preds = preds[:, 0]
    preds = np.where(preds > 0.5, 1, 0)
    y_test = y_test["Survived"]
    report = classification_report(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    kappa = cohen_kappa_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    percent_survived = np.sum(y_test) / len(y_test)
    print("Percent survived:", percent_survived)
    print("Total accuracy:", accuracy)
    print("Kappa:", kappa)
    print("Classification report:")
    print(report)
    
    
    # Predictions
    only_take_cols=X.columns.values.tolist() + ["PassengerId"]
    only_take_cols.remove("Survived")
    print(f"X.columns: {only_take_cols}")
    df_test = load_data("Titanic/titanic/test.csv", only_take_cols=only_take_cols, drop_non_num_cols=False)
    # Drop rows with a missing Survived column
    df_test.fillna(-1, inplace=True)
    # Add Survived column to the first column
    df_test.insert(0, "Survived", -1)
    pass_id = df_test["PassengerId"]
    df_test.drop(["PassengerId"], axis=1, inplace=True)
    # Ensure df_test is in the same order as X_train
    df_test = df_test[X.columns]
    X_test = df_test
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    preds = y_pred[:, 0]
    preds = np.where(preds > 0.5, 1, 0)
    # Output predictions
    df_out = pd.DataFrame({"PassengerId": pass_id, "Survived": preds})
    df_out.to_csv("Titanic/titanic/submission.csv", index=False)
    
    
