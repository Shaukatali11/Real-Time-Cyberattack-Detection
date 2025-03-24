import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier




# Define all supervised ML models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=42),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}



# Initialize result storage
results = []



# Train and test each model
for model_name, model in models.items():
    # Train the model and record training time
    start_time = time.time()
    model.fit(X_train, y_train_enc)
    end_time = time.time()
    
    # Predict and record detection time
    detection_start = time.time()
    predictions = model.predict(X_test)
    detection_end = time.time()
    
    # Calculate accuracy and detection time
    accuracy = accuracy_score(y_test_enc, predictions) * 100
    detection_time = detection_end - detection_start
    
    # Store the results
    results.append({
        "Model": model_name,
        "Accuracy (%)": round(accuracy, 2),
        "Detection Time (seconds)": round(detection_time, 4)
    })




# Create a DataFrame for results
results_df = pd.DataFrame(results)
print(results_df)

