# ================================================================
# CS5173 HW1 - Linear Regression and Neural Network Regression
# Cancer Mortality Prediction
# ================================================================
# Author: JOVAN JOSE ASKER FREDY
# Date: 09/20/2025
# ================================================================
# This script loads and preprocesses the cancer mortality dataset,
# trains a baseline Linear Regression model and several Deep Neural
# Network (DNN) architectures, and compares their performance using
# Mean Squared Error (MSE) and R² score.
# ================================================================

# -----------------------------
# Imported Libraries
# -----------------------------
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================================================================
# Step 0: Reproducibility
# ================================================================
# Set random seeds so results are reproducible across runs.
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ================================================================
# Step 1: Load & Preprocess Data
# ================================================================
# Load dataset (update path if necessary).
df = pd.read_csv("HW_1/cancer_reg-1.csv", encoding="latin-1")

# Drop non-numeric or redundant columns that don’t help prediction.
drop_cols = ["Geography", "binnedInc", "PctSomeCol18_24"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Remove rows where the target label is missing.
df = df.dropna(subset=["TARGET_deathRate"])

# Fill missing values in feature columns with the median of each column.
# (This prevents errors during training and is a safe imputation method.)
for col in df.columns:
    if col != "TARGET_deathRate":
        df[col] = df[col].fillna(df[col].median())

# Log-transform selected highly skewed numeric columns.
# log1p(x) = log(1+x) is used to handle zero values safely.
for col in ["popEst2015", "studyPerCap", "avgAnnCount", "avgDeathsPerYear"]:
    if col in df.columns:
        df[col] = np.log1p(df[col])

# Separate features (X) and target (y).
X = df.drop(columns=["TARGET_deathRate"])
y = df["TARGET_deathRate"]

# Scale features with RobustScaler (less sensitive to outliers).
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into train (70%), validation (15%), and test (15%).
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# ================================================================
# Step 1: Dataset Exploration (Answers)
# ================================================================
print("\n=== Step 1: Dataset Exploration ===")
print(f"1) Number of samples: {df.shape[0]}")
print("2) Problem: Predict cancer mortality rates (Regression).")
print(f"3) Min value in dataset: {df.min().min():.4f}, Max value: {df.max().max():.4f}")
print(f"4) Number of features per sample: {df.shape[1] - 1}")
print(f"5) Missing values present? {'Yes' if df.isnull().sum().sum() > 0 else 'No (handled with median imputation)'}")
print("6) Label: TARGET_deathRate")
print("7) Train/Val/Test Split: 70% / 15% / 15%")
print("8) Preprocessing: Dropped irrelevant cols, filled missing with median, log-transform skewed, scaled with RobustScaler.\n")

# ================================================================
# Step 2: Baseline Model (Linear Regression)
# ================================================================
results = {}

# Train Linear Regression on training data.
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate on validation set (baseline performance).
y_val_pred = lr.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

# Store results for later comparison.
results[("Linear Regression", "N/A")] = (mse_val, r2_val)

print(f"Linear Regression: MSE={mse_val:.4f}, R2={r2_val:.4f}")

# ================================================================
# Step 3: Build Deep Neural Network (Helper Function)
# ================================================================
def build_dnn(input_dim, layers_list, lr=0.001):
    """
    Build a fully connected DNN with given layer sizes.
    Args:
        input_dim (int): number of input features
        layers_list (list): hidden layer sizes, e.g., [30, 8]
        lr (float): learning rate
    Returns:
        keras.Model: compiled DNN model
    """
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers with ReLU activation, BatchNorm, and Dropout
    for units in layers_list:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())   # helps stabilize training
        model.add(layers.Dropout(0.1))           # reduces overfitting
    
    # Output layer (regression → linear activation)
    model.add(layers.Dense(1, activation="linear"))
    
    # Compile model with MSE loss and Adam optimizer
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="mse", metrics=["mse"])
    return model

# ================================================================
# Step 4: Train & Evaluate DNN Models
# ================================================================
# Define model architectures to test.
architectures = {
    "DNN-16": [16],
    "DNN-30-8": [30, 8],
    "DNN-30-16-8": [30, 16, 8],
    "DNN-30-16-8-4": [30, 16, 8, 4],
    "DNN-32-16-8": [32, 16, 8],  
}

# Learning rates to test for each model
lrs = [0.1, 0.01, 0.001, 0.0001]

# Directory for saving plots
os.makedirs("plots", exist_ok=True)

# Train each architecture × learning rate combination
for name, arch in architectures.items():
    for lr_val in lrs:
        # Build model
        model = build_dnn(X_train.shape[1], arch, lr=lr_val)
        
        # Early stopping: stop training if val loss doesn’t improve
        es = keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3000, batch_size=128, verbose=0, callbacks=[es]
        )
        
        # Evaluate on test set
        y_pred = model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save results
        results[(name, lr_val)] = (mse, r2)
        print("%s (LR=%.4f): MSE=%.4f, R2=%.4f" % (name, lr_val, mse, r2))
        
        # Save model to disk
        model.save(f"{name}_LR{lr_val}.keras")
        
        # Save loss curve plot
        plt.figure()
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title(f"{name} (LR={lr_val}) Training")
        plt.legend()
        plt.savefig(f"plots/{name}_LR{lr_val}_loss.png")
        plt.close()

# ================================================================
# Step 5: Report Best Model
# ================================================================
# Find model with highest R² on test set
best_model = max(results.items(), key=lambda kv: kv[1][1])
print("\n=== BEST MODEL ===")
print("Architecture=%s, LR=%s, MSE=%.4f, R2=%.4f" % (
    best_model[0][0], best_model[0][1], best_model[1][0], best_model[1][1]
))

# ================================================================
# Step 6 Part I: Results Table
# ================================================================
# Convert results dictionary into DataFrame
df_results = pd.DataFrame([
    {"Model": k[0], "LR": k[1], "MSE": v[0], "R2": v[1]}
    for k, v in results.items()
])

# Sort by R² for easier comparison
df_results = df_results.sort_values(by="R2", ascending=False).reset_index(drop=True)
print("\n=== All Model Results (sorted by R²) ===")
print(df_results)

# Save results to CSV
df_results.to_csv("model_results.csv", index=False)

# ================================================================
# Step 6 Part II: Best DNN vs Baseline Plot
# ================================================================
# Identify best DNN (exclude Linear Regression row)
best_dnn_row = df_results[df_results["Model"] != "Linear Regression"].iloc[0]
best_dnn_name = best_dnn_row["Model"]
best_dnn_r2 = best_dnn_row["R2"]

# Get Linear Regression R²
lr_row = df_results[df_results["Model"] == "Linear Regression"].iloc[0]
lr_r2 = lr_row["R2"]

# Bar chart: Linear Regression vs Best DNN
plt.figure(figsize=(6, 5))
plt.bar(["Linear Regression", best_dnn_name], [lr_r2, best_dnn_r2], color=["gray", "skyblue"])
plt.ylabel("R² Score")
plt.title("Best DNN vs Linear Regression (Baseline)")
for i, v in enumerate([lr_r2, best_dnn_r2]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
plt.ylim(0, 1)
plt.savefig("plots/best_vs_baseline.png")
plt.show()

# ================================================================
# Step 7: Test Model
# ================================================================
def test_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    Args:
        model: trained Keras or sklearn model
        X_test: test features
        y_test: true test labels
    Returns:
        mse, r2
    """
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Handle potential NaNs or Infs
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test Results → MSE={mse:.4f}, R²={r2:.4f}")
    return mse, r2