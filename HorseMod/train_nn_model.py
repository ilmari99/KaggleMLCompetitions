""" Train a model, that takes in a horse medical vector, with masked values, and outputs the same vector, with the masked values filled in.
"""

import numpy as np
import pandas as pd
from load_data import modify_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
# tf addons
import tensorflow_addons as tfa
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

class CustomMSE(tf.keras.losses.Loss):
    """ Custom loss function, that calculates MSE between predictions and true values,
    if the true value is not -1.
    """
    def __init__(self, name="custom_mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # The one-hot-encoded results are the first three columns
        # Apply softmax to the first three columns
        cat_cross_ent_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true[:,:3], y_pred[:,:3])
        y_true = y_true[:,3:]
        y_pred = y_pred[:,3:]
        mask = tf.math.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        mse_loss = tf.keras.losses.MSE(y_true, y_pred) * 0.001
        return cat_cross_ent_loss + mse_loss
    
class CustomAccuracy(tf.keras.metrics.Metric):
    """ Custom accuracy metric, that calculates accuracy between predictions and true values.
    
    y_pred is a vector of shape (batch_size, 28), where the first three columns are the scores for the classificatio
    y_true is a vector of shape (batch_size, 28), where the first three columns are the one-hot-encoded classification
    """
    
    def __init__(self, name="custom_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="accuracy", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # The one-hot-encoded results are the first three columns
        # Apply softmax to the first three columns
        y_pred = y_pred[:,:3]
        y_true = y_true[:,:3]
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_true, axis=1)
        # Calculate accuracy
        corrects = tf.cast(tf.equal(y_pred, y_true), tf.float32)
        accuracy = tf.reduce_sum(corrects) / tf.cast(tf.shape(y_pred)[0], tf.float32)
        self.accuracy.assign_add(accuracy)
        self.total.assign_add(1)
        
    def result(self):
        return self.accuracy / self.total
    
    def reset_states(self):
        self.accuracy.assign(0)
        self.total.assign(0)


def get_model(input_shape):
    """ Create a model that takes in a vector, with masked values, and outputs the same vector, with the masked values filled in. """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(input_shape[0], activation="linear"),
    ])
    model.compile(optimizer="adam", loss=CustomMSE(), metrics=[CustomAccuracy()])
    return model

if __name__ == "__main__":
    df = pd.read_csv("HorseMod/train.csv")
    # Add horse.csv data. In this file, all values are in double quotes
    horse_df = pd.read_csv("HorseMod/horse.csv")
    df = pd.concat([df, horse_df], axis=0)
    print(df.head())
    df = df.sample(frac=1)
    df = modify_data(df,get_columns=60)
    df.rename(columns={"outcome_outcome_died": "outcome_died", "outcome_outcome_euthanized": "outcome_euthanized", "outcome_outcome_lived": "outcome_lived"}, inplace=True)
    print(df.head())
    
    # Convert dtypes to float32
    df = df.astype(np.float32)
    # Split data into train and test sets
    X_train, X_test = train_test_split(df, test_size=0.1, random_state=42)
    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # Mask 25% of X_train values
    X_train_mask = np.random.choice([True, False], size=X_train.shape, p=[0.05, 0.95])
    X_train_masked = X_train.copy()
    X_train_masked[X_train_mask] = -1
    # Mask the outcome columns (0-2) of X_train
    X_train_masked[:,:3] = 0
    
    # Mask only the outcome columns (0-2) of X_test
    X_test_masked = X_test.copy()
    X_test_masked[:, :3] = 0

    print(f"Dtypes: {X_train_masked.dtype}, {X_test_masked.dtype}, {X_train.dtype}, {X_test.dtype}")
    print(X_train_masked.shape, X_test_masked.shape)
    
    # Create model
    model = get_model(X_train_masked.shape[1:])
    model.summary()
    # Train model
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True, mode="min")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="HorseMod/logs", histogram_freq=1)
    model.fit(X_train_masked, X_train, epochs=3000, validation_data=(X_test_masked, X_test), batch_size=32, callbacks=[early_stop, tensorboard])

    # Calculate accuracy on test set
    y_full_pred = model.predict(X_test_masked)
    y_pred = tf.nn.softmax(y_full_pred[:, :3]).numpy()
    y_pred = y_pred.argmax(axis=1)
    
    y_true = X_test[:, :3].argmax(axis=1)
    
    # Calculate accuracy
    accuracy = np.sum(y_pred == y_true) / len(y_pred)
    print(f"Accuracy: {accuracy}")
    f1 = f1_score(y_true, y_pred, average="micro")
    print(f"F1 Score: {f1}")
    
    # Write to submission file
    df_test = pd.read_csv("HorseMod/test.csv")
    horse_ids = df_test["id"]
    train_cols = df.columns.values.tolist()
    train_cols.remove("outcome_died")
    train_cols.remove("outcome_euthanized")
    train_cols.remove("outcome_lived")
    print(train_cols)
    df_test = modify_data(df_test,get_columns=train_cols)
    # Make sure the columns are in the same order as in df
    df_test = df_test[train_cols]
    
    test_arr = df_test.to_numpy()
    # Add three columns of 0s to test_arr
    test_arr = np.hstack((test_arr, np.zeros((test_arr.shape[0], 3))))
    
    preds = model.predict(test_arr)
    preds = preds.argmax(axis=1)
    # Convert preds to {0: died, 1: euthanized, 2: lived}
    preds = pd.DataFrame({"id": horse_ids, "outcome": preds.flatten()})
    preds["outcome"].replace({0: "died", 1: "euthanized", 2: "lived"}, inplace=True)
    df_submission = pd.DataFrame({"id": horse_ids, "outcome": preds["outcome"]})
    print(df_submission.head())
    ans = input("Save submission? (y/n): ")
    df_submission.to_csv("HorseMod/submission.csv", index=False)
    
    
    






