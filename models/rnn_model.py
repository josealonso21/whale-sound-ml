import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def train_rnn(X_train, y_train, batch_size, epochs, lstm_units, dense_units, dropout_rate, spatial_dropout, learning_rate):
    if len(X_train.shape) == 2:
        X_train = X_train[..., np.newaxis]

    model = Sequential([
        Bidirectional(LSTM(lstm_units[0], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
        SpatialDropout1D(spatial_dropout),
        BatchNormalization(),
        Bidirectional(LSTM(lstm_units[1], return_sequences=True)),
        SpatialDropout1D(spatial_dropout),
        BatchNormalization(),
        Bidirectional(LSTM(lstm_units[2])),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Nadam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weights
    )

    y_val_pred = (model.predict(X_val) > 0.5).astype("int32")

    print("Precision:", precision_score(y_val, y_val_pred))
    print("Recall:", recall_score(y_val, y_val_pred))
    print("F1-Score:", f1_score(y_val, y_val_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

    return model