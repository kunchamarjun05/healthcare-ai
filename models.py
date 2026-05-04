"""
Healthcare AI — Model Training & Evaluation Module
Trains Random Forest, XGBoost, and SVM models for:
  1. Heart Disease Prediction
  2. Diabetes Prediction
  3. Breast Cancer Detection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.datasets import load_breast_cancer
# Using sklearn GradientBoosting instead of XGBoost for compatibility
import warnings
warnings.filterwarnings('ignore')


# ============================================================
#  DATA LOADERS
# ============================================================

def load_heart_data():
    """Load the Cleveland Heart Disease dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    try:
        df = pd.read_csv(url, names=columns, na_values='?')
        df.dropna(inplace=True)
        df['target'] = (df['target'] > 0).astype(int)  # Binary: 0 = no disease, 1 = disease
        return df
    except Exception:
        # Fallback: generate synthetic heart data
        return _generate_heart_data()


def _generate_heart_data():
    """Generate synthetic heart disease data as fallback."""
    np.random.seed(42)
    n = 303
    data = {
        'age': np.random.randint(29, 77, n),
        'sex': np.random.choice([0, 1], n, p=[0.32, 0.68]),
        'cp': np.random.choice([0, 1, 2, 3], n),
        'trestbps': np.random.randint(94, 200, n),
        'chol': np.random.randint(126, 564, n),
        'fbs': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'restecg': np.random.choice([0, 1, 2], n),
        'thalach': np.random.randint(71, 202, n),
        'exang': np.random.choice([0, 1], n, p=[0.67, 0.33]),
        'oldpeak': np.round(np.random.uniform(0, 6.2, n), 1),
        'slope': np.random.choice([0, 1, 2], n),
        'ca': np.random.choice([0, 1, 2, 3], n),
        'thal': np.random.choice([0, 1, 2, 3], n),
    }
    df = pd.DataFrame(data)
    # Create somewhat realistic target
    risk = (df['age'] > 55).astype(int) + (df['cp'] >= 2).astype(int) + (df['thalach'] < 140).astype(int)
    df['target'] = (risk >= 2).astype(int)
    noise = np.random.random(n) < 0.15
    df.loc[noise, 'target'] = 1 - df.loc[noise, 'target']
    return df


def load_diabetes_data():
    """Load Pima Indians Diabetes dataset."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    try:
        df = pd.read_csv(url, names=columns)
        return df
    except Exception:
        return _generate_diabetes_data()


def _generate_diabetes_data():
    """Generate synthetic diabetes data as fallback."""
    np.random.seed(42)
    n = 768
    data = {
        'Pregnancies': np.random.randint(0, 17, n),
        'Glucose': np.random.randint(44, 199, n),
        'BloodPressure': np.random.randint(24, 122, n),
        'SkinThickness': np.random.randint(0, 99, n),
        'Insulin': np.random.randint(0, 846, n),
        'BMI': np.round(np.random.uniform(18, 67, n), 1),
        'DiabetesPedigreeFunction': np.round(np.random.uniform(0.08, 2.42, n), 3),
        'Age': np.random.randint(21, 81, n),
    }
    df = pd.DataFrame(data)
    risk = (df['Glucose'] > 140).astype(int) + (df['BMI'] > 30).astype(int) + (df['Age'] > 40).astype(int)
    df['Outcome'] = (risk >= 2).astype(int)
    return df


def load_cancer_data():
    """Load sklearn Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    # In sklearn: 0 = malignant, 1 = benign. We flip for intuition: 1 = cancer detected
    df['target'] = 1 - df['target']
    return df, data.feature_names.tolist()


# ============================================================
#  MODEL TRAINING
# ============================================================

MODELS = {
    'Random Forest': lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': lambda: SVC(kernel='rbf', probability=True, random_state=42),
    'Logistic Regression': lambda: LogisticRegression(max_iter=1000, random_state=42),
}


def train_and_evaluate(X, y, test_size=0.2, models_to_use=None):
    """
    Train multiple models, return results dict.
    
    Returns:
        dict with keys for each model name containing:
            - model: trained model
            - scaler: fitted scaler
            - accuracy, precision, recall, f1
            - y_test, y_pred, y_prob
            - cv_scores
            - confusion_matrix
            - roc_data (fpr, tpr, auc)
    """
    if models_to_use is None:
        models_to_use = list(MODELS.keys())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for name in models_to_use:
        if name not in MODELS:
            continue

        model = MODELS[name]()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

        # ROC
        roc_data = None
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_data = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

        # Feature importance (if available)
        feat_importance = None
        if hasattr(model, 'feature_importances_'):
            feat_importance = model.feature_importances_

        results[name] = {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_data': roc_data,
            'feature_importance': feat_importance,
        }

    return results, scaler


def predict_single(model, scaler, features):
    """Predict for a single input."""
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None
    return prediction, probability
