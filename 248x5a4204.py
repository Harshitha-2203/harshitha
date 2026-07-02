import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. GENERATE DUMMY DATA (Simulating 'x_bal' and 'y_bal' for loan approval)
# This ensures the code runs immediately without a missing file error.
np.random.seed(42)
num_samples = 1000

# Simulating 5 input features (e.g., Credit Score, Income, Debt, Loan Amount, Age)
x_bal = np.random.rand(num_samples, 5) 

# Simulating a binary target variable (0 = Denied, 1 = Approved)
y_bal = np.random.randint(0, 2, size=num_samples)

print(f"Generated x_bal shape: {x_bal.shape}")
print(f"Generated y_bal shape: {y_bal.shape}\n")


# 2. SPLIT THE DATA (Exactly as shown in your classroom assignment)
X_train, X_test, y_train, y_test = train_test_split(
    x_bal, 
    y_bal, 
    test_size=0.33, 
    random_state=42
)

# Print shapes to verify the split dimensions
print("--- Data Split Sizes ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}\n")


# 3. INITIALIZE & TRAIN THE MACHINE LEARNING MODEL
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
print("Model training completed successfully!\n")


# 4. PREDICT & EVALUATE
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"--- Final Evaluation ---")
print(f"Test Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
