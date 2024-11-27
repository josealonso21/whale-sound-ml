from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def train_svm(X, y, param_grid=None, test_size=0.2, random_state=42):
    if param_grid is None:
        param_grid = {
            'C': [1.3, 2, 2.5],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3]
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    randomized_search = RandomizedSearchCV(
        SVC(probability=True),
        param_grid,
        n_iter=10, #20
        cv=5,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1,
        random_state=random_state
    )
    randomized_search.fit(X_train, y_train)
    svm_model = randomized_search.best_estimator_

    y_val_pred = svm_model.predict(X_val)

    print("Best Parameters:", randomized_search.best_params_)
    print("SVM Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Precision:", precision_score(y_val, y_val_pred))
    print("Recall:", recall_score(y_val, y_val_pred))
    print("F1-Score:", f1_score(y_val, y_val_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

    return svm_model, scaler