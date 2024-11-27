import numpy as np
import joblib
import tensorflow as tf
from scripts.test_prediction import test_models
from scripts.data_preprocessing import preprocess_data
from models.rnn_model import train_rnn
from models.svm_model import train_svm
from models.cnn_model import train_cnn
from models.ensemble_model import combine_predictions, train_meta_model, evaluate_ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def main():
    # Preprocessing the data
    if os.path.exists('outputs/features.npy') and os.path.exists('outputs/labels.npy'):
        print("Loading preprocessed data...")
        features = np.load('outputs/features.npy')
        labels = np.load('outputs/labels.npy')
    else:
        print("Preprocessing data...")
        features, labels = preprocess_data('data/train', 'data/train.csv')
        np.save('outputs/features.npy',features)
        np.save('outputs/labels.npy',labels)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train RNN model
    if os.path.exists('outputs/rnn_model.h5'):
        print("Loading RNN model...")
        rnn_model = tf.keras.models.load_model('outputs/rnn_model.h5')
    else:
        print("Training RNN...")
        rnn_model = train_rnn(X_train, y_train, batch_size=64, epochs=30, lstm_units=(128, 64, 32), dense_units=64,
                          dropout_rate=0.4, spatial_dropout=0.3, learning_rate=0.001)
        rnn_model.save('outputs/rnn_model.h5')
    rnn_preds = rnn_model.predict(X_val[..., np.newaxis]).squeeze()

    # Train SVM model
    if os.path.exists('outputs/svm_model.pkl') and os.path.exists('outputs/svm_scaler.pkl'):
        print("Loading SVM model...")
        svm_model = joblib.load('outputs/svm_model.pkl')
        scaler = joblib.load('outputs/svm_scaler.pkl')
    else:
        print("Training SVM...")
        svm_model, scaler = train_svm(X_train, y_train)
        joblib.dump(svm_model, 'outputs/svm_model.pkl')
        joblib.dump(scaler, 'outputs/svm_scaler.pkl')
    svm_preds = svm_model.predict(scaler.transform(X_val))

    # Train CNN model
    if os.path.exists('outputs/cnn_model.h5'):
        print("Loading CNN model...")
        cnn_model = tf.keras.models.load_model('outputs/cnn_model.h5')
    else:
        print("Training CNN...")
        cnn_model = train_cnn(X_train, y_train, batch_size=64, epochs=40, num_filters=256, kernel_size=5, pool_size=2,
                          dense_units=128, dropout_rate=0.5, n_conv_layers=3, learning_rate=0.0001)
        cnn_model.save('outputs/cnn_model.h5')
    cnn_preds = cnn_model.predict(X_val[..., np.newaxis]).flatten()

    # Ensemble training
    print("Training Ensemble...")
    meta_model = train_meta_model(cnn_preds, rnn_preds, svm_preds, y_val)

    # Testing and evaluating models
    print("Testing Models...")
    test_features, test_labels = preprocess_data('data/test', 'data/sample_submission.csv', save_features=False)

    # Loading the trained models
    print("Loading models...")
    rnn_model = tf.keras.models.load_model('outputs/rnn_model.h5')
    svm_model = joblib.load('outputs/svm_model.pkl')
    scaler = joblib.load('outputs/svm_scaler.pkl')
    cnn_model = tf.keras.models.load_model('outputs/cnn_model.h5')

    # Making predictions on test data
    rnn_test_preds = rnn_model.predict(test_features[..., np.newaxis]).squeeze()
    cnn_test_preds = cnn_model.predict(test_features[..., np.newaxis]).squeeze()
    svm_test_preds = svm_model.predict(scaler.transform(test_features))

    # Train ensemble model
    print("Training Ensemble...")
    meta_model, scaler = train_meta_model(cnn_preds, rnn_preds, svm_preds, y_val)

    # Evaluate ensemble
    print("Evaluating Ensemble...")
    final_preds = evaluate_ensemble(meta_model, scaler, cnn_test_preds, rnn_test_preds, svm_test_preds, test_labels)

    # Optionally, print classification report of ensemble
    print("Classification Report for Ensemble Model:")
    print(classification_report(test_labels, final_preds))

if __name__ == "__main__":
    main()