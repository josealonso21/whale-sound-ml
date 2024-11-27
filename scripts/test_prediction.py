import numpy as np
import joblib
import tensorflow as tf
from scripts.data_preprocessing import preprocess_data

def load_model(model_path, model_type, scaler_path=None):
    if model_type == "svm":
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path else None
        return model, scaler
    elif model_type in ["cnn", "rnn"]:
        return tf.keras.models.load_model(model_path), None
    else:
        raise ValueError("Unsupported model type!")

def evaluate_model(model, scaler, features, labels, model_type):
    if model_type == "svm" and scaler:
        features = scaler.transform(features)
        preds = model.predict(features)
        # Convert hard labels to probabilities (0 or 1 to mimic probabilities)
        preds = preds.astype(float)
    elif model_type in ["cnn", "rnn"]:
        features = features[..., np.newaxis]  # Ensure correct input shape for CNN/RNN
        preds = model.predict(features).flatten()
    else:
        raise ValueError("Unsupported model type!")

    return preds

def test_models(test_dir, labels_path):
    print("Preprocessing test data...")
    test_features, test_labels = preprocess_data(test_dir, labels_path, save_features=False)

    print("Loading models...")
    svm_model, svm_scaler = load_model('outputs/svm_model.pkl', 'svm', 'outputs/svm_scaler.pkl')
    cnn_model, _ = load_model('outputs/cnn_model.h5', 'cnn')
    rnn_model, _ = load_model('outputs/rnn_model.h5', 'rnn')

    print("Evaluating SVM...")
    svm_preds = evaluate_model(svm_model, svm_scaler, test_features, test_labels, 'svm')

    print("Evaluating CNN...")
    cnn_preds = evaluate_model(cnn_model, None, test_features, test_labels, 'cnn')

    print("Evaluating RNN...")
    rnn_preds = evaluate_model(rnn_model, None, test_features, test_labels, 'rnn')

    return svm_preds, cnn_preds, rnn_preds, test_labels