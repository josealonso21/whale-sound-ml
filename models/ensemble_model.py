import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def combine_predictions(cnn_preds, rnn_preds, svm_preds, strategy='hard'):
    """Combine model predictions using hard or soft voting."""
    if strategy == 'hard':
        combined = np.round((cnn_preds + rnn_preds + svm_preds) / 3).astype(int)
    elif strategy == 'soft':
        # Normalize to probabilities if not already
        combined = np.mean([cnn_preds, rnn_preds, svm_preds], axis=0)
        combined = (combined > 0.5).astype(int)
    else:
        raise ValueError("Invalid strategy! Choose 'hard' or 'soft'.")
    return combined

def train_meta_model(cnn_preds, rnn_preds, svm_preds, y_train):
    """Train a meta-classifier (e.g., Logistic Regression) on model predictions."""
    # Standardize predictions for meta-model training
    meta_features = np.column_stack([cnn_preds, rnn_preds, svm_preds])
    scaler = StandardScaler()
    meta_features_scaled = scaler.fit_transform(meta_features)
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_features_scaled, y_train)
    return meta_model, scaler

def evaluate_ensemble(meta_model, scaler, cnn_preds, rnn_preds, svm_preds, y_test):
    """Evaluate the ensemble using a meta-model."""
    meta_features = np.column_stack([cnn_preds, rnn_preds, svm_preds])
    meta_features_scaled = scaler.transform(meta_features)
    final_preds = meta_model.predict(meta_features_scaled)
    print("Ensemble Performance:")
    print(classification_report(y_test, final_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))
    return final_preds