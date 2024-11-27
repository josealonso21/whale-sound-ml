import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def build_cnn(input_shape, num_filters=128, kernel_size=5, pool_size=2, dense_units=64, dropout_rate=0.5, n_conv_layers=3):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    for i in range(n_conv_layers):
        model.add(Conv1D(filters=num_filters, kernel_size=2, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_cnn(X, y, batch_size=64, epochs=30, learning_rate=0.001, patience=5, num_filters=64, kernel_size=3, pool_size=2, dense_units=64, dropout_rate=0.4, n_conv_layers=3):
    X = X[...,np.newaxis]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))

    model = build_cnn(input_shape=(X_train.shape[1], X_train.shape[2]),
                      num_filters=num_filters,
                      kernel_size=kernel_size,
                      pool_size=pool_size,
                      dense_units=dense_units,
                      dropout_rate=dropout_rate,
                      n_conv_layers=n_conv_layers)

    optimizer = Nadam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )

    y_val_pred = (model.predict(X_val) > 0.5).astype("int32")
    val_loss, val_accuracy = model.evaluate(X_val, y_val)

    print(f"CNN Validation Accuracy: {val_accuracy}")
    print("Precision:", precision_score(y_val, y_val_pred))
    print("Recall:", recall_score(y_val, y_val_pred))
    print("F1-Score:", f1_score(y_val, y_val_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    return model