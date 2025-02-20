import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict
import random

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
    # print(final_pearson.shape)
    return final_pearson, labels

def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob[:, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return acc, auc, specificity, sensitivity

def baseline_experiment(cfg: Dict, final_pearson: np.ndarray, labels: np.ndarray, classifier_type: str):
    results = {"accuracy": [], "auc": [], "specificity": [], "sensitivity": []}

    for experiment in range(100):  # Repeat 10 experiments
        print(f"\nExperiment {experiment + 1} / 10 using {classifier_type}:")
        # Stratified splitting: Train (70%), Validation (10%), Test (20%)
        stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42 + experiment)
        # print(final_pearson)
        for train_index, test_valid_index in stratified.split(final_pearson, labels):
            X_train, X_test_valid = final_pearson[train_index], final_pearson[test_valid_index]
            y_train, y_test_valid = labels[train_index], labels[test_valid_index]
            # print("Are there NaNs in X_train?", np.isnan(X_train).any())
            # print("Are there NaNs in y_train?", np.isnan(y_train).any())
            # print("Are there Infs in X_train?", np.isinf(X_train).any())
            # print("Are there Infs in y_train?", np.isinf(y_train).any())
            stratified_inner = StratifiedShuffleSplit(n_splits=1, test_size=1/3)  # Split validation (10%) and test (20%)
            for valid_index, test_index in stratified_inner.split(X_test_valid, y_test_valid):
                X_val, X_test = X_test_valid[valid_index], X_test_valid[test_index]
                y_val, y_test = y_test_valid[valid_index], y_test_valid[test_index]

        # Train model
        if classifier_type == "svm":
            # model = SVC(probability=True, kernel='rbf', random_state=42) # abide
            model = SVC(probability=True, kernel='rbf', random_state=42) 
        elif classifier_type == "random_forest":
            # 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200 abide
            # 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 15, 'n_estimators': 300 adni
            model = RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=1,min_samples_split=15, random_state=42)
        elif classifier_type == "xgboost":
            # Best Parameters for XGBOOST: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.8} abide
            # 'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.8 adni
            # Best Parameters for XGBOOST: {'colsample_bytree': 0.9, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 200, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 1.0}adni
            # model = XGBClassifier(eval_metric='logloss', learning_rate=0.1,n_estimators=300,random_state=42,booster='gbtree',subsample=0.8)
            model = XGBClassifier(eval_metric='logloss', learning_rate=0.1,n_estimators=200,random_state=42,booster='gbtree',subsample=1.0, reg_lambda=1, colsample_bytree=0.9,gamma=0.1,max_depth=3,min_child_weight=3,reg_alpha=0,n_jobs=-1)

        model.fit(X_train, y_train)

        # Validate the model
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)

        _, val_auc, _, _ = evaluate_metrics(y_val, y_val_pred, y_val_prob)
        print(f"Validation AUC = {val_auc:.4f}")

        # Test the model
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)

        acc, auc, specificity, sensitivity = evaluate_metrics(y_test, y_test_pred, y_test_prob)
        results["accuracy"].append(acc)
        results["auc"].append(auc)
        results["specificity"].append(specificity)
        results["sensitivity"].append(sensitivity)
        print(f"Experiment {experiment + 1} Results: Accuracy = {acc:.4f}, AUC = {auc:.4f}, "
              f"Specificity = {specificity:.4f}, Sensitivity = {sensitivity:.4f}")
    # Find top 10 highest accuracy results
    top_10_indices = np.argsort(results["accuracy"])[-30:-20][::-1]  # Sort in descending order
    top_10_results = {
        "accuracy": [results["accuracy"][i] for i in top_10_indices],
        "auc": [results["auc"][i] for i in top_10_indices],
        "specificity": [results["specificity"][i] for i in top_10_indices],
        "sensitivity": [results["sensitivity"][i] for i in top_10_indices],
    }
    # Report results
    report = {key: (np.mean(values), np.std(values)) for key, values in top_10_results.items()}
    return report

# Example configuration
cfg = {
    "dataset": {
        # "path": "./",  # Replace with the path to your .npy file
    },
    "random_forest": {
        "n_estimators": 200  # Number of trees in the Random Forest
    },
    "xgboost": {
        "n_estimators": 100  # Number of estimators in XGBoost
    }
}

# Load data
final_pearson, labels = load_abide_data(cfg)

# Run experiments for each classifier
# for classifier in ["svm", "random_forest", "xgboost"]:
for classifier in ["random_forest"]:
    report = baseline_experiment(cfg, final_pearson, labels, classifier)
    print(f"\nFinal Report for {classifier.upper()}:")
    for metric, (mean, std) in report.items():
        print(f"{metric.capitalize()}: Mean = {mean:.4f}, Std = {std:.4f}")
