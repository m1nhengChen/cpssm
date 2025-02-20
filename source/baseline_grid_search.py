import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict
import random
import gc
# export LOKY_TIMEOUT=3600

class StandardScaler:
    """
    Standardize the input
    """

    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def transform(self, data: np.array):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.array):
        return (data * self.std) + self.mean

def load_abide_data(cfg: Dict):
    data = np.load(cfg['dataset']['path'], allow_pickle=True).item()
    final_pearson = data["corr"]
    labels = data["label"]
    # final_timeseires = data["timeseires"]
    # scaler = StandardScaler(mean=np.mean(
    #     final_timeseires), std=np.std(final_timeseires))
    final_pearson = final_pearson.reshape(final_pearson.shape[0], -1)  # Flatten last two dimensions

    return final_pearson, labels

def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob[:, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return acc, auc, specificity, sensitivity

def grid_search(cfg: Dict, final_pearson: np.ndarray, labels: np.ndarray, classifier_type: str):
    print(f"\nRunning Grid Search for {classifier_type.upper()}")
    # Define parameter grid
    if classifier_type == "random_forest":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [5, 5, 15],
            "min_samples_leaf": [1, 2, 4]
        }
        model = RandomForestClassifier(random_state=42)
    elif classifier_type == "xgboost":
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],       # Step size shrinkage used to prevent overfitting. Range: [0,1]. Typical values: 0.01-0.2. :contentReference[oaicite:0]{index=0}
            'n_estimators': [ 200, 300],          # Number of boosting rounds. More trees can improve performance but may lead to overfitting. Range: 100 to 1000. :contentReference[oaicite:1]{index=1}
            'max_depth': [3, 5, 7],                   # Maximum depth of a tree. Increasing it makes the model more complex and likely to overfit. Typical values: 3-10. :contentReference[oaicite:2]{index=2}
            'min_child_weight': [1, 3, 5],            # Minimum sum of instance weight (hessian) needed in a child. Larger values prevent overfitting. :contentReference[oaicite:3]{index=3}
            'subsample': [0.8, 0.9, 1.0],             # Subsample ratio of the training instances. Lower values prevent overfitting. Range: 0.5-1. :contentReference[oaicite:4]{index=4}
            'colsample_bytree': [0.8, 0.9, 1.0],      # Subsample ratio of columns when constructing each tree. Lower values prevent overfitting. Range: 0.5-1. :contentReference[oaicite:5]{index=5}
            'gamma': [0, 0.1, 0.2],                   # Minimum loss reduction required to make a split. Larger values make the algorithm more conservative. Range: [0,∞]. :contentReference[oaicite:6]{index=6}
            'reg_alpha': [0, 0.01, 0.1],              # L1 regularization term on weights. Increasing this value will make model more conservative. Range: [0,∞]. :contentReference[oaicite:7]{index=7}
            'reg_lambda': [1, 1.5, 2.0]               # L2 regularization term on weights. Increasing this value will make model more conservative. Range: [0,∞]. :contentReference[oaicite:8]{index=8}
}

        model = XGBClassifier(eval_metric="logloss", random_state=42,n_jobs=-1)

    # Custom scorer for AUC
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "auc": make_scorer(roc_auc_score, needs_proba=True)
    }

    # Stratified split
    stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in stratified.split(final_pearson, labels):
        X_train, X_test = final_pearson[train_index], final_pearson[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    # Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,  # 3-fold cross-validation
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Best parameters and evaluation
    print(f"Best Parameters for {classifier_type.upper()}: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # Test the model
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)

    acc, auc, specificity, sensitivity = evaluate_metrics(y_test, y_test_pred, y_test_prob)
    print(f"Test Results for {classifier_type.upper()}:")
    print(f"Accuracy = {acc:.4f}, AUC = {auc:.4f}, Specificity = {specificity:.4f}, Sensitivity = {sensitivity:.4f}")
    gc.collect()

# Example configuration
cfg = {
    "dataset": {
        "path": "./",  # Replace with the path to your .npy file
    }
}

# Load data
final_pearson, labels = load_abide_data(cfg)

# Run grid search for Random Forest and XGBoost
# grid_search(cfg, final_pearson, labels, "random_forest")
grid_search(cfg, final_pearson, labels, "xgboost")
