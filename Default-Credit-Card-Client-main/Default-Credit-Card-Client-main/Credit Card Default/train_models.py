import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle

# Load data
 # Using raw string to avoid escaping backslashes
 # Change this to your dataset path
data = pd.read_csv("C:\\Credit Card Default\\UCI_Credit_Card.csv")


# Clean and preprocess data
data = data.dropna()  # Drop missing values
X = data.drop(columns=['ID', 'default.payment.next.month'])  # Features
y = data['default.payment.next.month']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Save models
with open('random_forest_model.pkl', 'wb') as rf_file:
    pickle.dump(rf_model, rf_file)

with open('log_regression_model.pkl', 'wb') as log_reg_file:
    pickle.dump(log_reg_model, log_reg_file)

# Model Evaluation
rf_pred = rf_model.predict(X_test)
log_reg_pred = log_reg_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)

print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Logistic Regression Accuracy: {log_reg_accuracy}")
