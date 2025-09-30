# ================================================================
# CS5173 HW2 - Convolutional Neural Networks (CNN)
# Handwritten Digit Classification (MNIST)
# ================================================================
# Author: JOVAN JOSE ASKER FREDY
# Date: 09/23/2025
# ================================================================
# This script loads and preprocesses the MNIST dataset,
# trains a baseline Logistic Regression classifier and several
# Convolutional Neural Network (CNN) architectures, and compares
# their performance using Accuracy, F1 score, and ROC-AUC.
# ================================================================

# -----------------------------
# Imported Libraries
# -----------------------------
import os
import joblib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50

os.makedirs("plots", exist_ok=True)

# ================================================================
# Step 0: Reproducibility
# ================================================================
SEED = 42
np.random.seed(SEED)      # set numpy seed
random.seed(SEED)         # set python random seed
tf.random.set_seed(SEED)  # set TensorFlow seed

# ================================================================
# Step 1: Load & Preprocess Data
# ================================================================
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values from range [0,255] → [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a channel dimension (grayscale → shape (28,28,1))
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode labels for neural networks
lb = LabelBinarizer()
y_train_oh = lb.fit_transform(y_train)
y_test_oh = lb.transform(y_test)

# Split into Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    x_train, y_train, test_size=0.3, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED
)

# Match splits for one-hot labels
_, y_train_oh = train_test_split(y_train_oh, test_size=0.3, random_state=SEED)
y_val_oh, y_test_oh = train_test_split(lb.transform(y_temp), test_size=0.5, random_state=SEED)

# ================================================================
# Step 1: Dataset Exploration (Answers)
# ================================================================
print("\n=== Step 1: Dataset Exploration ===")
print(f"1) Number of samples: {x_train.shape[0] + x_test.shape[0]}") # total = 70k
print("2) Problem: Classify handwritten digits (0–9).")
print(f"3) Min value in dataset: {x_train.min():.4f}, Max value: {x_train.max():.4f}")
print(f"4) Number of features per sample: {28*28}") # flattened pixels
print(f"5) Missing values present? No")
print("6) Label: Digit class (0–9)")
print("7) Train/Val/Test Split: 70% / 15% / 15%")
print("8) Preprocessing: Normalized pixels to [0,1], added channel dim, one-hot labels.\n")

# ================================================================
# Step 2: Baseline Model (Logistic Regression)
# ================================================================
results = {}

# Flatten images into 1D vectors for logistic regression
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_val_flat = X_val.reshape((X_val.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Train logistic regression model
lr = LogisticRegression(max_iter=1000, solver="lbfgs")
lr.fit(X_train_flat, y_train)

# Predict on validation set
y_val_pred = lr.predict(X_val_flat)
acc_val = accuracy_score(y_val, y_val_pred)
f1_val = f1_score(y_val, y_val_pred, average="macro")

# ROC-AUC requires predicted probabilities
y_val_proba = lr.predict_proba(X_val_flat)
auc_val = roc_auc_score(pd.get_dummies(y_val), y_val_proba, multi_class="ovr")

# Store baseline results
results[("Logistic Regression", "N/A")] = (acc_val, f1_val, auc_val)
print(f"Logistic Regression: Acc={acc_val:.4f}, F1={f1_val:.4f}, AUC={auc_val:.4f}")

# ================================================================
# Step 3: Build Models (Helper Functions)
# ================================================================

# Custom F1 score metric for Keras
def f1_metric(y_true, y_pred):
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=10) # convert to one-hot
    y_pred_bin = K.cast(K.greater(y_pred, 0.5), K.floatx())          # threshold predictions

    # compute TP, FP, FN
    tp = K.sum(y_true_one_hot * y_pred_bin, axis=0)
    fp = K.sum((1 - y_true_one_hot) * y_pred_bin, axis=0)
    fn = K.sum(y_true_one_hot * (1 - y_pred_bin), axis=0)

    # precision, recall, and F1
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())

    return K.mean(f1)

# ---------------- DNN ----------------
def build_dnn(input_shape, hidden_layers=[128, 64], lr=0.001):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Flatten()) # flatten image
    for units in hidden_layers: # add hidden layers
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation="softmax")) # output layer
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", f1_metric])
    return model

# ---------------- ConvNet ----------------
def build_convnet(input_shape, lr=0.001):
    model = keras.Sequential([
        keras.Input(shape=input_shape),  
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", f1_metric])
    return model

# ---------------- VGG-like ----------------
def build_vgg(input_shape, lr=0.001):
    model = keras.Sequential([
        keras.Input(shape=input_shape),  
        # two conv layers per block (VGG-style)
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", f1_metric]
    )
    return model

# ---------------- ResNet ----------------
# Define a residual convolutional block
def conv_block(x, filters, stride=1):
    shortcut = x  # identity shortcut
    # first conv layer
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # second conv layer
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # adjust shortcut if dimensions mismatch
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # add residual connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Build a ResNet-like architecture
def build_resnet(input_shape, lr=0.001):
    inputs = keras.Input(shape=input_shape)

    # Resize to 32x32 for ResNet structure
    x = layers.Resizing(32, 32)(inputs)
    x = layers.Conv2D(16, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 3 stages with residual blocks
    for filters, blocks, stride in [(16, 2, 1), (32, 2, 2), (64, 2, 2)]:
        for i in range(blocks):
            s = stride if i == 0 else 1
            x = conv_block(x, filters, stride=s)

    # global pooling and dense output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", f1_metric]
    )
    return model


# ================================================================
# Step 4: Train & Evaluate Models
# ================================================================
# dictionary of architectures to test
architectures = {
    "DNN": lambda: build_dnn((28,28,1)),
    "ConvNet": lambda: build_convnet((28,28,1)),
    "VGG": lambda: build_vgg((28,28,1)),
    "ResNet": lambda: build_resnet((28,28,1))
}

results = {}   # store evaluation metrics
histories = {} # store training histories

# Train each architecture with different learning rates
for name, builder in architectures.items():
    for lr_val in [0.1, 0.01, 0.001]:
        print(f"\nTraining {name} (LR={lr_val})...")
        model = builder()
        es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, batch_size=128,
            verbose=0, callbacks=[es]
        )
        histories[(name, lr_val)] = history

        # Predict + compute metrics
        y_pred = np.argmax(model.predict(X_test), axis=1)
        f1 = f1_score(y_test, y_pred, average="macro")
        y_probs = model.predict(X_test)
        auc = roc_auc_score(tf.keras.utils.to_categorical(y_test, 10), y_probs, multi_class="ovr")

        # store results
        results[(name, lr_val)] = (f1, auc)
        print(f"{name} (LR={lr_val}): F1={f1:.4f}, AUC={auc:.4f}")

        # Save training/validation curves
        plt.figure()
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.plot(history.history["accuracy"], label="Train Acc")
        plt.plot(history.history["val_accuracy"], label="Val Acc")
        plt.title(f"{name} (LR={lr_val}) Training")
        plt.xlabel("Epochs"); plt.ylabel("Value")
        plt.legend(); plt.savefig(f"plots/{name}_LR{lr_val}_training.png")
        plt.close()

# ================================================================
# Step 6: Report Results + ROC Curve
# ================================================================
# Collect results into DataFrame
df_results = pd.DataFrame([
    {"Model": k[0], "LR": k[1], "F1": v[0], "AUC": v[1]}
    for k, v in results.items()
]).sort_values(by="F1", ascending=False).reset_index(drop=True)

print("\n=== All Model Results (sorted by F1) ===")
print(df_results)

# Train best model on combined train+val, plot ROC curve
best_row = df_results.iloc[0]
best_model_name, best_lr = best_row["Model"], best_row["LR"]
best_model = architectures[best_model_name]()  # rebuild
best_model.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]),
               epochs=50, batch_size=128, verbose=0)
y_probs = best_model.predict(X_test)
fpr, tpr, _ = roc_curve(tf.keras.utils.to_categorical(y_test, 10).ravel(), y_probs.ravel())

plt.figure()
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={best_row['AUC']:.4f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Best Model)")
plt.legend(); plt.savefig("plots/best_model_roc.png")
plt.show()

# ================================================================
# Step 7: Test Model
# ================================================================
def test_model(model, X_test, y_test):
    """Evaluate trained model on test data."""
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(pd.get_dummies(y_test), model.predict(X_test, verbose=0), multi_class="ovr")
    print(f"Test Results → Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    return acc, f1, auc
