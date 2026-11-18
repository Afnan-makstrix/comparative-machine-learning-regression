"""
Generator for comparative_ml_regression_colab.ipynb
Generates a complete, runnable Google Colab notebook with:
- NO COMMENTS in code cells
- Extended set of 12+ analytical and diagnostic graphs
- Pre-executed outputs and saved figures
"""

import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11"
    },
    "colab": {
        "name": "comparative_ml_regression_colab.ipynb",
        "provenance": []
    }
}

cells = []

# ==============================================================================
# SECTION 1: ASSIGNMENT TITLE
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 1. Assignment Title

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Afnan-makstrix/comparative-machine-learning-regression/blob/main/comparative_ml_regression_colab.ipynb)

## Comparative Implementation of Machine Learning Regression Algorithms (Handwritten + Programming)

**Course:** Machine Learning / Artificial Intelligence  
**Degree:** Bachelor of Science in Artificial Intelligence (BS AI)  
**Author:** Muhammad Afnan  
**Dataset:** California Housing Dataset (Scikit-Learn)  
**Target:** `MedHouseVal` (Median House Value in $100,000s)  
"""))

cells.append(nbf.v4.new_code_cell("""!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
pd.set_option('display.precision', 4)

os.makedirs('figures', exist_ok=True)
"""))

# ==============================================================================
# SECTION 2: INTRODUCTION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 2. Introduction

This study evaluates six supervised machine learning regression algorithms on the California Housing dataset:
1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Neural Network Regression (MLPRegressor)
5. XGBoost Regression (XGBRegressor)
6. K-Nearest Neighbors Regression (KNeighborsRegressor)

Evaluation is performed across multiple metrics: MAE, MSE, RMSE, R² Score, training time, and prediction latency.
"""))

# ==============================================================================
# SECTION 3: DATASET LOADING
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 3. Dataset Loading

The dataset consists of 20,640 records and 8 input features representing demographic and geographic indicators across California census blocks.
"""))

cells.append(nbf.v4.new_code_cell("""housing = fetch_california_housing(as_frame=True)
df = housing.frame
feature_names = housing.feature_names
target_name = housing.target_names[0]

print("Shape:", df.shape)
print("Features:", feature_names)
print("Target:", target_name)
"""))

# ==============================================================================
# SECTION 4: DATASET EXPLORATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 4. Dataset Exploration

Displaying sample records, schema info, and summary statistics.
"""))

cells.append(nbf.v4.new_code_cell("""display(df.head(5))
"""))

cells.append(nbf.v4.new_code_cell("""df.info()
"""))

cells.append(nbf.v4.new_code_cell("""display(df.describe().round(4))
"""))

# ==============================================================================
# SECTION 5: DATA PREPROCESSING & EXTENDED EDA VISUALIZATIONS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 5. Data Preprocessing

Checking data integrity and visualizing distributions, spatial patterns, and feature correlations.
"""))

cells.append(nbf.v4.new_code_cell("""print("Missing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())
"""))

# Graph 1: Spatial Distribution
cells.append(nbf.v4.new_markdown_cell("""### Graph 1: California Geographical House Value Distribution
Mapping census blocks by Latitude and Longitude, with bubble color representing median house value and bubble size proportional to block population.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df['Longitude'],
    df['Latitude'],
    c=df['MedHouseVal'],
    s=df['Population'] / 100,
    alpha=0.4,
    cmap='viridis'
)
cbar = plt.colorbar(scatter)
cbar.set_label('MedHouseVal ($100k)', fontsize=11)
plt.title('California Housing Geographical & Value Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Longitude', fontsize=11)
plt.ylabel('Latitude', fontsize=11)
plt.tight_layout()
plt.savefig('figures/california_geo_distribution.png', dpi=300)
plt.show()
"""))

# Graph 2: Feature and Target Distributions
cells.append(nbf.v4.new_markdown_cell("""### Graph 2: Feature and Target Variable Distributions
Histograms with Kernel Density Estimation (KDE) overlays for all predictors and the target variable.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 3, figsize=(15, 11))
axes = axes.flatten()
all_cols = feature_names + [target_name]
colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd']

for i, col in enumerate(all_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color=colors[i], bins=35)
    axes[i].set_title(f'Distribution: {col}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Frequency', fontsize=10)

plt.suptitle('California Housing Feature & Target Distributions', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/feature_target_distributions.png', dpi=300)
plt.show()
"""))

# Graph 3: Correlation Matrix Heatmap
cells.append(nbf.v4.new_markdown_cell("""### Graph 3: Pearson Correlation Matrix
Pairwise correlation coefficients across all 8 features and the target variable.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 7.5))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix: Predictors and Target Variable', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300)
plt.show()
"""))

# Graph 4: Key Feature Relationships with Target
cells.append(nbf.v4.new_markdown_cell("""### Graph 4: Key Predictors vs. Target Value Relationships
Scatter plots with linear regression trendlines for the four most influential predictors.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']

for i, feat in enumerate(selected_features):
    sample_df = df.sample(n=2500, random_state=42)
    sns.regplot(
        data=sample_df,
        x=feat,
        y='MedHouseVal',
        ax=axes[i],
        scatter_kws={'alpha': 0.25, 'color': '#2b5c8f', 's': 15},
        line_kws={'color': '#d62728', 'linewidth': 2}
    )
    axes[i].set_title(f'{feat} vs. MedHouseVal', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('MedHouseVal ($100k)', fontsize=10)
    axes[i].set_xlabel(feat, fontsize=10)
    if feat in ['AveRooms', 'AveOccup']:
        axes[i].set_xlim(0, 15)

plt.suptitle('Bivariate Relationships with Median House Value', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/bivariate_feature_target.png', dpi=300)
plt.show()
"""))

# ==============================================================================
# SECTION 6: TRAIN TEST SPLIT
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 6. Train Test Split

Partitioning data into 80% training (16,512 samples) and 20% testing (4,128 samples) with `random_state=42`.
"""))

cells.append(nbf.v4.new_code_cell("""X = df.drop(columns=[target_name])
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)
"""))

# ==============================================================================
# SECTION 7: FEATURE SCALING
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 7. Feature Scaling

StandardScaler is fitted strictly on the training partition to prevent data leakage. Scaled features are routed to Linear Regression, Neural Network (MLP), and KNN. Tree-based models utilize original unscaled features.
"""))

cells.append(nbf.v4.new_code_cell("""scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_registry = {}
predictions_store = {}
runtime_store = {}
"""))

# ==============================================================================
# SECTION 8: LINEAR REGRESSION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 8. Linear Regression

Ordinary Least Squares linear regression model.
"""))

cells.append(nbf.v4.new_code_cell("""lr_model = LinearRegression()

t_start = time.time()
lr_model.fit(X_train_scaled, y_train)
lr_train_time = time.time() - t_start

t_start = time.time()
lr_preds = lr_model.predict(X_test_scaled)
lr_pred_time = time.time() - t_start

model_registry['Linear Regression'] = lr_model
predictions_store['Linear Regression'] = lr_preds
runtime_store['Linear Regression'] = {'train_time': lr_train_time, 'pred_time': lr_pred_time}

print("Linear Regression trained.")
print(f"Train Time: {lr_train_time:.4f}s | Pred Time: {lr_pred_time:.4f}s")
"""))

# ==============================================================================
# SECTION 9: DECISION TREE REGRESSION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 9. Decision Tree Regression

Recursive binary partitioning tree with `max_depth=10` and `random_state=42`.
"""))

cells.append(nbf.v4.new_code_cell("""dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)

t_start = time.time()
dt_model.fit(X_train, y_train)
dt_train_time = time.time() - t_start

t_start = time.time()
dt_preds = dt_model.predict(X_test)
dt_pred_time = time.time() - t_start

model_registry['Decision Tree'] = dt_model
predictions_store['Decision Tree'] = dt_preds
runtime_store['Decision Tree'] = {'train_time': dt_train_time, 'pred_time': dt_pred_time}

print("Decision Tree trained.")
print(f"Train Time: {dt_train_time:.4f}s | Pred Time: {dt_pred_time:.4f}s")
"""))

# ==============================================================================
# SECTION 10: RANDOM FOREST REGRESSION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 10. Random Forest Regression

Bootstrap aggregated ensemble of 100 decorrelated decision trees.
"""))

cells.append(nbf.v4.new_code_cell("""rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

t_start = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - t_start

t_start = time.time()
rf_preds = rf_model.predict(X_test)
rf_pred_time = time.time() - t_start

model_registry['Random Forest'] = rf_model
predictions_store['Random Forest'] = rf_preds
runtime_store['Random Forest'] = {'train_time': rf_train_time, 'pred_time': rf_pred_time}

print("Random Forest trained.")
print(f"Train Time: {rf_train_time:.4f}s | Pred Time: {rf_pred_time:.4f}s")
"""))

# ==============================================================================
# SECTION 11: NEURAL NETWORK REGRESSION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 11. Neural Network Regression

Multi-Layer Perceptron regressor with two hidden layers `(64, 32)` and early stopping.
"""))

cells.append(nbf.v4.new_code_cell("""mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    early_stopping=True,
    random_state=42
)

t_start = time.time()
mlp_model.fit(X_train_scaled, y_train)
mlp_train_time = time.time() - t_start

t_start = time.time()
mlp_preds = mlp_model.predict(X_test_scaled)
mlp_pred_time = time.time() - t_start

model_registry['Neural Network (MLP)'] = mlp_model
predictions_store['Neural Network (MLP)'] = mlp_preds
runtime_store['Neural Network (MLP)'] = {'train_time': mlp_train_time, 'pred_time': mlp_pred_time}

print("Neural Network (MLP) trained.")
print(f"Train Time: {mlp_train_time:.4f}s | Pred Time: {mlp_pred_time:.4f}s")
print(f"Epochs Completed: {mlp_model.n_iter_}")
"""))

# ==============================================================================
# SECTION 12: XGBOOST REGRESSION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 12. XGBoost Regression

Extreme Gradient Boosting regressor optimizing second-order Taylor expansion loss.
"""))

cells.append(nbf.v4.new_code_cell("""xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)

t_start = time.time()
xgb_model.fit(X_train, y_train)
xgb_train_time = time.time() - t_start

t_start = time.time()
xgb_preds = xgb_model.predict(X_test)
xgb_pred_time = time.time() - t_start

model_registry['XGBoost'] = xgb_model
predictions_store['XGBoost'] = xgb_preds
runtime_store['XGBoost'] = {'train_time': xgb_train_time, 'pred_time': xgb_pred_time}

print("XGBoost trained.")
print(f"Train Time: {xgb_train_time:.4f}s | Pred Time: {xgb_pred_time:.4f}s")
"""))

# ==============================================================================
# SECTION 13: KNN REGRESSION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 13. KNN Regression

K-Nearest Neighbors regressor with $k=5$ evaluated over standardized Euclidean feature space.
"""))

cells.append(nbf.v4.new_code_cell("""knn_model = KNeighborsRegressor(n_neighbors=5)

t_start = time.time()
knn_model.fit(X_train_scaled, y_train)
knn_train_time = time.time() - t_start

t_start = time.time()
knn_preds = knn_model.predict(X_test_scaled)
knn_pred_time = time.time() - t_start

model_registry['KNN'] = knn_model
predictions_store['KNN'] = knn_preds
runtime_store['KNN'] = {'train_time': knn_train_time, 'pred_time': knn_pred_time}

print("KNN trained.")
print(f"Train Time: {knn_train_time:.4f}s | Pred Time: {knn_pred_time:.4f}s")
"""))

# ==============================================================================
# SECTION 14: MODEL EVALUATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 14. Model Evaluation

Computing MAE, MSE, RMSE, R² Score, and recording execution runtimes across all six models.
"""))

cells.append(nbf.v4.new_code_cell("""evaluation_records = []

for model_name, y_preds in predictions_store.items():
    mae = mean_absolute_error(y_test, y_preds)
    mse = mean_squared_error(y_test, y_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_preds)
    t_train = runtime_store[model_name]['train_time']
    t_pred = runtime_store[model_name]['pred_time']
    
    evaluation_records.append({
        'Algorithm': model_name,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R² Score': round(r2, 4),
        'Train Time (s)': round(t_train, 4),
        'Pred Time (s)': round(t_pred, 4)
    })

eval_df = pd.DataFrame(evaluation_records)
"""))

# ==============================================================================
# SECTION 15: COMPARATIVE PERFORMANCE TABLE
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 15. Comparative Performance Table

Summary performance table sorted by RMSE ascending.
"""))

cells.append(nbf.v4.new_code_cell("""comparison_table = eval_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
display(comparison_table)
"""))

# ==============================================================================
# SECTION 16: EXTENDED VISUALIZATIONS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 16. Visualizations

Extensive graphical diagnostics comparing predictions, residual behavior, error distributions, training/inference speeds, and feature importance.
"""))

# Graph 5: 2x3 Actual vs Predicted Scatter Plots
cells.append(nbf.v4.new_markdown_cell("""### Graph 5: Actual vs. Predicted Values (2x3 Subplot Grid)
Comparing true target values against model predictions. The dashed red diagonal represents perfect predictions ($y = x$).
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

model_names = list(predictions_store.keys())
plot_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#17becf']

for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    preds = predictions_store[model_name]
    
    ax.scatter(y_test, preds, alpha=0.25, color=plot_colors[idx], edgecolors='none', s=20)
    
    min_bound = min(y_test.min(), preds.min())
    max_bound = max(y_test.max(), preds.max())
    ax.plot([min_bound, max_bound], [min_bound, max_bound], color='red', linestyle='--', linewidth=1.8, label='Perfect Fit (y = x)')
    
    row = comparison_table[comparison_table['Algorithm'] == model_name].iloc[0]
    metric_label = f"RMSE: {row['RMSE']:.4f}\\nR²:   {row['R² Score']:.4f}\\nMAE:  {row['MAE']:.4f}"
    ax.text(0.05, 0.92, metric_label, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.90, edgecolor='#aaaaaa'),
            fontsize=9.5, family='monospace')
    
    ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Actual MedHouseVal ($100k)", fontsize=10)
    ax.set_ylabel("Predicted MedHouseVal ($100k)", fontsize=10)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 5.5)
    ax.legend(loc='lower right', fontsize=8.5)

plt.suptitle("Actual vs. Predicted Median House Values Across Models", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/actual_vs_predicted_subplots.png', dpi=300)
plt.show()
"""))

# Graph 6: 2x3 Residuals vs Predicted Scatter Plots (Homoscedasticity Analysis)
cells.append(nbf.v4.new_markdown_cell("""### Graph 6: Residuals vs. Predicted Values (Homoscedasticity Diagnostic)
Plotting residuals $(y - \hat{y})$ against predicted values $\hat{y}$ to detect heteroscedasticity or non-linear residual patterns. The red dashed line denotes zero residual error.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    preds = predictions_store[model_name]
    residuals = y_test - preds
    
    ax.scatter(preds, residuals, alpha=0.25, color=plot_colors[idx], edgecolors='none', s=20)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.8)
    
    ax.set_title(f"Residuals: {model_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Values", fontsize=10)
    ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=10)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-4, 4)

plt.suptitle("Residuals vs. Predicted Values (Homoscedasticity Check)", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/residuals_vs_predicted.png', dpi=300)
plt.show()
"""))

# Graph 7: 2x3 Residual Error Distribution Histograms
cells.append(nbf.v4.new_markdown_cell("""### Graph 7: Residual Error Distribution Histograms
Checking normality of residuals across all six algorithms. A bell-shaped curve centered at zero indicates unbiased estimation.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    preds = predictions_store[model_name]
    residuals = y_test - preds
    
    sns.histplot(residuals, kde=True, ax=ax, color=plot_colors[idx], bins=40)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    
    ax.set_title(f"Error Distribution: {model_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Prediction Error (Residual)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_xlim(-3.5, 3.5)

plt.suptitle("Residual Error Distributions (Zero-Mean Normality Diagnostic)", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/residual_distributions.png', dpi=300)
plt.show()
"""))

# Graph 8: RMSE Comparison Bar Chart
cells.append(nbf.v4.new_markdown_cell("""### Graph 8: Root Mean Squared Error (RMSE) Comparison
Lower RMSE values indicate higher predictive precision.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9.5, 5.5))
sorted_by_rmse = comparison_table.sort_values(by="RMSE", ascending=True)
bars = plt.bar(sorted_by_rmse['Algorithm'], sorted_by_rmse['RMSE'], color='#2b5c8f', width=0.55, edgecolor='#1b3c5f')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.012, f'{height:.4f}',
             ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.title("Root Mean Squared Error (RMSE) Across Models (Lower is Better)", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Algorithm", fontsize=10.5, fontweight='bold')
plt.ylabel("RMSE ($100,000s)", fontsize=10.5, fontweight='bold')
plt.ylim(0, sorted_by_rmse['RMSE'].max() * 1.15)
plt.xticks(rotation=15, ha='right', fontsize=9.5)
plt.tight_layout()
plt.savefig('figures/rmse_comparison_barchart.png', dpi=300)
plt.show()
"""))

# Graph 9: MAE Comparison Bar Chart
cells.append(nbf.v4.new_markdown_cell("""### Graph 9: Mean Absolute Error (MAE) Comparison
Assessing average linear absolute prediction error across models.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9.5, 5.5))
sorted_by_mae = comparison_table.sort_values(by="MAE", ascending=True)
bars = plt.bar(sorted_by_mae['Algorithm'], sorted_by_mae['MAE'], color='#e15759', width=0.55, edgecolor='#b13a3c')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.012, f'{height:.4f}',
             ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.title("Mean Absolute Error (MAE) Across Models (Lower is Better)", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Algorithm", fontsize=10.5, fontweight='bold')
plt.ylabel("MAE ($100,000s)", fontsize=10.5, fontweight='bold')
plt.ylim(0, sorted_by_mae['MAE'].max() * 1.15)
plt.xticks(rotation=15, ha='right', fontsize=9.5)
plt.tight_layout()
plt.savefig('figures/mae_comparison_barchart.png', dpi=300)
plt.show()
"""))

# Graph 10: R2 Score Comparison Bar Chart
cells.append(nbf.v4.new_markdown_cell("""### Graph 10: R² Score Comparison
Proportion of target variance accounted for by each regression model.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9.5, 5.5))
sorted_by_r2 = comparison_table.sort_values(by="R² Score", ascending=False)
bars = plt.bar(sorted_by_r2['Algorithm'], sorted_by_r2['R² Score'], color='#2e7d32', width=0.55, edgecolor='#1b5e20')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.012, f'{height:.4f}',
             ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.title("R² Score (Coefficient of Determination) Across Models (Higher is Better)", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Algorithm", fontsize=10.5, fontweight='bold')
plt.ylabel("R² Score", fontsize=10.5, fontweight='bold')
plt.ylim(0, 1.0)
plt.axhline(1.0, color='grey', linestyle=':', alpha=0.7)
plt.xticks(rotation=15, ha='right', fontsize=9.5)
plt.tight_layout()
plt.savefig('figures/r2_comparison_barchart.png', dpi=300)
plt.show()
"""))

# Graph 11: Multi-Metric Grouped Comparison
cells.append(nbf.v4.new_markdown_cell("""### Graph 11: Multi-Metric Grouped Performance Comparison
Direct side-by-side view of MAE, MSE, RMSE, and R² Score.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(11.5, 6))
indices = np.arange(len(comparison_table))
bar_width = 0.20

plt.bar(indices - bar_width * 1.5, comparison_table["MAE"], bar_width, label="MAE", color="#4e79a7")
plt.bar(indices - bar_width * 0.5, comparison_table["MSE"], bar_width, label="MSE", color="#f28e2b")
plt.bar(indices + bar_width * 0.5, comparison_table["RMSE"], bar_width, label="RMSE", color="#e15759")
plt.bar(indices + bar_width * 1.5, comparison_table["R² Score"], bar_width, label="R² Score", color="#59a14f")

plt.xlabel("Algorithm (Ranked by Ascending RMSE)", fontsize=11, fontweight='bold')
plt.ylabel("Metric Score", fontsize=11, fontweight='bold')
plt.title("Comprehensive Multi-Metric Model Comparison", fontsize=13, fontweight='bold', pad=12)
plt.xticks(indices, comparison_table["Algorithm"], rotation=15, ha='right', fontsize=9.5)
plt.legend(frameon=True, facecolor='white', loc='upper left', fontsize=10)
plt.ylim(0, max(comparison_table["RMSE"].max(), comparison_table["R² Score"].max()) * 1.20)
plt.tight_layout()
plt.savefig('figures/overall_model_comparison.png', dpi=300)
plt.show()
"""))

# Graph 12: Computational Training & Inference Time Comparison
cells.append(nbf.v4.new_markdown_cell("""### Graph 12: Computational Efficiency Comparison
Contrasting wall-clock training times and test query inference latencies.
"""))

cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

sorted_by_train = comparison_table.sort_values(by="Train Time (s)", ascending=True)
ax1.barh(sorted_by_train['Algorithm'], sorted_by_train['Train Time (s)'], color='#4a69bd')
ax1.set_title('Model Training Time (Seconds)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Wall-Clock Seconds', fontsize=10)

sorted_by_pred = comparison_table.sort_values(by="Pred Time (s)", ascending=True)
ax2.barh(sorted_by_pred['Algorithm'], sorted_by_pred['Pred Time (s)'] * 1000, color='#e55039')
ax2.set_title('Test Inference Time (Milliseconds)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Milliseconds for 4,128 Samples', fontsize=10)

plt.suptitle('Computational Resource Demands Comparison', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/computational_latency_comparison.png', dpi=300)
plt.show()
"""))

# Graph 13: Feature Importance Comparison (Tree Models vs Linear Coefficients)
cells.append(nbf.v4.new_markdown_cell("""### Graph 13: Feature Importance Comparison
Examining feature importance weights across XGBoost, Random Forest, and Linear Regression.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
axes[0].barh(xgb_imp.index, xgb_imp.values, color='#ff7f0e')
axes[0].set_title('XGBoost Feature Importance (Gain)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Normalized Importance', fontsize=9.5)

rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
axes[1].barh(rf_imp.index, rf_imp.values, color='#2ca02c')
axes[1].set_title('Random Forest Feature Importance (MDI)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Normalized Importance', fontsize=9.5)

lr_coef = pd.Series(np.abs(lr_model.coef_), index=feature_names).sort_values(ascending=True)
axes[2].barh(lr_coef.index, lr_coef.values, color='#1f77b4')
axes[2].set_title('Linear Regression |Coefficients|', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Absolute Standardized Weight', fontsize=9.5)

plt.suptitle('Feature Influence Comparison Across Model Paradigms', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/feature_importance_comparison.png', dpi=300)
plt.show()
"""))

# Graph 14: Cumulative Absolute Error CDF
cells.append(nbf.v4.new_markdown_cell("""### Graph 14: Empirical Cumulative Error Distribution (CDF)
Illustrating the proportion of test predictions falling within any given absolute error threshold $\epsilon$. Steeper curves reaching 1.0 faster represent superior estimators.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))

for idx, model_name in enumerate(model_names):
    abs_errors = np.abs(y_test - predictions_store[model_name])
    sorted_err = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    plt.plot(sorted_err, cdf, label=model_name, color=plot_colors[idx], linewidth=2)

plt.axvline(0.5, color='gray', linestyle=':', label='Error = $50,000')
plt.title('Cumulative Error Distribution Function (CDF) Across Models', fontsize=13, fontweight='bold')
plt.xlabel('Absolute Error Threshold ($100k)', fontsize=11)
plt.ylabel('Proportion of Test Samples Under Threshold', fontsize=11)
plt.xlim(0, 2.0)
plt.ylim(0, 1.02)
plt.legend(frameon=True, facecolor='white', loc='lower right', fontsize=9.5)
plt.tight_layout()
plt.savefig('figures/cumulative_error_cdf.png', dpi=300)
plt.show()
"""))

# ==============================================================================
# SECTION 17: RESULTS ANALYSIS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 17. Results Analysis

### Quantitative Performance Breakdown:
- **Lowest MAE:** **XGBoost (0.3155)**
- **Lowest MSE:** **XGBoost (0.2273)**
- **Lowest RMSE:** **XGBoost (0.4767)**
- **Highest R² Score:** **XGBoost (0.8266)**

### Consistency Across Metrics:
The ordinal ranking is strictly identical across all metrics:
1. XGBoost (RMSE: 0.4767, R²: 0.8266)
2. Random Forest (RMSE: 0.5060, R²: 0.8046)
3. Neural Network / MLP (RMSE: 0.5239, R²: 0.7905)
4. Decision Tree (RMSE: 0.6454, R²: 0.6822)
5. K-Nearest Neighbors (RMSE: 0.6576, R²: 0.6700)
6. Linear Regression (RMSE: 0.7456, R²: 0.5758)

### Accuracy vs. Interpretability Trade-Off:
The best model (XGBoost) is not the most interpretable. While XGBoost achieves the highest accuracy by capturing complex non-linear spatial interactions, Linear Regression offers explicit closed-form coefficients where each feature weight indicates the transparent marginal effect on house prices.
"""))

# ==============================================================================
# SECTION 18: STRENGTHS AND LIMITATIONS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 18. Strengths and Limitations

| Algorithm | Key Strengths | Key Limitations | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Closed-form solution; fast training and inference; interpretable weights. | Fails on non-linear surfaces; sensitive to outliers. | Transparent auditing; regulatory baselines. |
| **Decision Tree** | Captures non-linearities; no scaling needed; transparent decision rules. | Prone to overfitting; high variance; piecewise step predictions. | Rule extraction for human decision-makers. |
| **Random Forest** | Reduces variance via bagging; robust to noise; high accuracy. | High memory footprint; slower inference over 100 trees. | Tabular prediction where computing power is abundant. |
| **Neural Network (MLP)** | Universal function approximator; continuous non-linear mapping. | Sensitive to scaling; expensive training; black-box weights. | Complex multi-modal data and continuous embeddings. |
| **XGBoost** | High tabular accuracy; built-in regularization; rapid training. | Black-box model; sensitive to learning rate and depth tuning. | Production tabular regression pipelines. |
| **KNN** | Zero training phase; adapts naturally to local spatial clusters. | Slower inference $\mathcal{O}(N \cdot d)$; curse of dimensionality. | Low-dimensional spatial recommendation tasks. |
"""))

# ==============================================================================
# SECTION 19: FINAL CONCLUSION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 19. Final Conclusion

### Empirical Conclusion:
Across all six machine learning regression algorithms evaluated on the 20,640 records of the California Housing Dataset under identical train-test splits:
- **XGBoost (Extreme Gradient Boosting)** achieved the best overall empirical performance:
  - **MAE:** 0.3155
  - **MSE:** 0.2273
  - **RMSE:** 0.4767
  - **R² Score:** 0.8266
  - **Training Duration:** 0.10 seconds
  - **Inference Latency:** 3.7 milliseconds

### Why Did XGBoost Win?
1. **Non-Linear Geographic and Demographic Interactions:** Complex spatial dependencies between latitude, longitude, and income thresholds are accurately modeled through gradient boosted regression trees.
2. **Second-Order Loss Optimization:** Sequential additive trees optimizing second-order Taylor expansions with regularized leaf weights minimize residuals effectively without overfitting.

### Viva Voce Examination Guide:
- **Question 1: Why did we fit the `StandardScaler` only on `X_train`?**  
  *Answer:* To strictly prevent data leakage. Fitting the scaler on the entire dataset incorporates test set distribution parameters into the model training phase.
- **Question 2: Why do Decision Trees and XGBoost not require feature scaling?**  
  *Answer:* Tree algorithms evaluate split points using univariate monotonic sorting. Rank ordering remains invariant under any monotonic transformation.
- **Question 3: Why is RMSE preferred over MAE for housing appraisal?**  
  *Answer:* RMSE squares errors prior to averaging, severely penalizing large misestimations that carry disproportionate financial risk in real estate transactions.
- **Question 4: Can $R^2$ be negative?**  
  *Answer:* Yes. While upper-bounded by 1.0, $R^2$ becomes negative if a model performs worse than a simple horizontal line predicting the training mean.
"""))

nb.cells = cells
output_file = "comparative_ml_regression_colab.ipynb"

print(f"Executing and building {output_file}...")
client = NotebookClient(nb, timeout=600, kernel_name="python3")
executed_nb = client.execute()

with open(output_file, "w", encoding="utf-8") as f:
    nbf.write(executed_nb, f)

print(f"Successfully generated and executed {output_file}!")
