# CS-5173 HW1: Linear Regression and Neural Network Regression 

## 📌 Project Overview
This homework compares a baseline **Linear Regression** model with several **Deep Neural Network (DNN)** architectures for predicting **Cancer Mortality Rates** (`TARGET_deathRate`).  

The pipeline follows the **six machine learning training components**:
1. Data Exploration  
2. Model Design  
3. Objective Function  
4. Optimization  
5. Model Selection  
6. Performance Evaluation  

---

## 📝 Step 1: Dataset Exploration

Dataset file: `cancer_reg-1.csv`

1. **Number of samples**: `3047`  
2. **Problem type**: Predict **cancer mortality rates** (Regression task).  
3. **Value range**: Minimum = `0.0000`, Maximum ≈ `12.5945` (after preprocessing).  
4. **Number of features per sample**: `29` (after dropping irrelevant columns).  
5. **Missing values**: Some present → handled using **median imputation**.  
6. **Label**: `TARGET_deathRate`.  
7. **Train/Val/Test split**: **70% / 15% / 15%**.  
8. **Preprocessing steps**:  
   - Dropped irrelevant columns (`Geography`, `binnedInc`, `PctSomeCol18_24`)  
   - Filled missing values with **median**  
   - Log-transform for skewed features (`popEst2015`, `studyPerCap`, `avgAnnCount`, `avgDeathsPerYear`)  
   - RobustScaler applied (handles outliers better)  

---

## 🏗 Step 2: Models

The following models were tested:

| Model             | Hidden Layers   | Activation | Dropout | BatchNorm | Learning Rates |
|-------------------|----------------|------------|---------|-----------|----------------|
| Linear Regression | N/A            | N/A        | N/A     | N/A       | Default (sklearn) |
| DNN-16            | [16]           | ReLU       | 0.1     | Yes       | 0.1, 0.01, 0.001, 0.0001 |
| DNN-30-8          | [30, 8]        | ReLU       | 0.1     | Yes       | 0.1, 0.01, 0.001, 0.0001 |
| DNN-30-16-8       | [30, 16, 8]    | ReLU       | 0.1     | Yes       | 0.1, 0.01, 0.001, 0.0001 |
| DNN-30-16-8-4     | [30, 16, 8, 4] | ReLU       | 0.1     | Yes       | 0.1, 0.01, 0.001, 0.0001 |
| DNN-32-16-8       | [32, 16, 8]    | ReLU       | 0.1     | Yes       | 0.1, 0.01, 0.001, 0.0001 |

---

## 🎯 Step 3: Objective
- **Loss Function**: Mean Squared Error (MSE)  

\[
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

---

## ⚡ Step 4: Optimization
- **Optimizer**: Adam (`learning_rate` varied as hyperparameter)  
- Early stopping with patience = 200 epochs  

---

## 🔎 Step 5: Model Selection

All models were trained with different learning rates.  

Example performance summary:

| Model          | LR     | MSE     | R²     |
|----------------|--------|---------|--------|
| Linear Regression | N/A    | 450.3198 | 0.4330 |
| DNN-16         | 0.001  | 312.5741 | 0.6212 |
| DNN-30-8       | 0.001  | 102.1675 | 0.8714 |
| DNN-30-16-8    | 0.001  | 118.2953 | 0.8547 |
| DNN-30-16-8-4  | 0.001  | 131.4598 | 0.8422 |
| DNN-32-16-8    | 0.001  | 127.3031 | 0.8469 |

---

## 📈 Step 6: Model Performance

1. **Best Model**:  
   - **Architecture**: `DNN-30-8`  
   - **Learning Rate**: `0.001`  
   - **MSE**: `102.1675`  
   - **R²**: `0.8714`  

2. **Comparison Plot**:  
   `plots/best_vs_baseline.png` shows R² of **Linear Regression vs Best DNN**.  

   ![Best vs Baseline](plots/best_vs_baseline.png)

3. **Training Loss Example**:  
   Each model has its own loss curve saved under `plots/`. Example:  

   ![DNN Training](plots/DNN-30-8_LR0.001_loss.png)

---

## ▶️ How to Run

### Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow joblib
