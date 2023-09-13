""" Train a model, that takes in a horse medical vector, with masked values, and outputs the same vector, with the masked values filled in.
"""

import numpy as np
import pandas as pd
from load_data import modify_data
from sklearn.model_selection import train_test_split
import tensorflow as tf

class CustomMSE(tf.keras.losses.Loss):
    """ Custom loss function, that calculates MSE between predictions and true values,
    if the true value is not -1.
    """
    def __init__(self, name="custom_mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mask = tf.math.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        mse_loss = tf.keras.losses.MSE(y_true, y_pred)
        return mse_loss

def get_model(input_shape):
    """ Create a model that takes in a vector, with masked values, and outputs the same vector, with the masked values filled in. """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=input_shape),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(input_shape[0], activation="linear"),
    ])
    model.compile(optimizer="adam", loss=CustomMSE(), metrics=["mae"])
    return model

if __name__ == "__main__":
    df = pd.read_csv("HorseMod/train.csv")
    df = modify_data(df)
    print(df.head())
    # Convert dtypes to float32
    df = df.astype(np.float32)

    # Split data into train and test sets
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    # Mask 30% of X_train and X_test values
    X_train_masked = X_train.mask(np.random.random(X_train.shape) < .2)
    X_test_masked = X_test.mask(np.random.random(X_test.shape) < .2)
    # Fill missing values with -1
    X_train_masked = X_train_masked.fillna(-1.0)
    X_test_masked = X_test_masked.fillna(-1.0)

    # Convert to numpy arrays
    X_train_masked = X_train_masked.to_numpy(dtype=np.float32)
    X_test_masked = X_test_masked.to_numpy(dtype=np.float32)
    #X_train_masked[:,-1] = -1
    #X_test_masked[:,-1] = -1
    X_train = X_train.to_numpy(dtype=np.float32)
    X_test = X_test.to_numpy(dtype=np.float32)

    print(f"Dtypes: {X_train_masked.dtype}, {X_test_masked.dtype}, {X_train.dtype}, {X_test.dtype}")
    print(X_train_masked.shape, X_test_masked.shape)
    
    # Create model
    model = get_model(X_train_masked.shape[1:])
    model.summary()
    # Train model
    model.fit(X_train_masked, X_train, epochs=300, validation_data=(X_test_masked, X_test), batch_size=128)

    # Convert all X_test['outcome'] values to -1
    X_test_pred = X_test.copy()
    X_test_pred[:,-1] = -1

    # Calculate accuracy on the 'outcome' column (last column)
    y_pred = model.predict(X_test_pred)
    y_pred = np.round(y_pred)
    y_pred = np.where(y_pred < 0, 0, y_pred)
    y_pred = np.where(y_pred > 3, 3, y_pred)
    y_pred = y_pred.astype(int)
    y_pred = y_pred[:,-1]
    y_test = X_test[:,-1]
    print(y_pred[:10], y_test[:10])
    print("Accuracy:", np.sum(y_pred == y_test) / len(y_test))






