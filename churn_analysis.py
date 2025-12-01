#-----------------------------------------
# CUSTOMER CHURN ANALYSIS & PREDICTION
#-----------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#-----------------------------------------
# 1. LOAD DATA
#-----------------------------------------
df = pd.read_csv("customer_churn.csv")
print("Data Loaded Successfully!")
print(df.head())


#-----------------------------------------
# 2. DATA PREPROCESSING
#-----------------------------------------

# Convert categorical to numeric
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Handle missing values
df = df.fillna(df.mean())

# Feature / Target split
X = df.drop("Churn", axis=1)
y = df["Churn"]


#-----------------------------------------
# 3. TRAIN-TEST SPLIT
#-----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#-----------------------------------------
# 4. TRAIN MODELS
#-----------------------------------------

# Logistic Regression
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)


#-----------------------------------------
# 5. ACCURACY & PERFORMANCE
#-----------------------------------------
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))


#-----------------------------------------
# 6. FEATURE IMPORTANCE
#-----------------------------------------
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_imp.nlargest(10).plot(kind='bar')
plt.title("Top 10 Feature Importance")
plt.show()


#-----------------------------------------
# 7. PREDICT NEW CUSTOMER
#-----------------------------------------

# Example customer (modify values as needed)
new_customer = pd.DataFrame({
    "Gender":[1],
    "SeniorCitizen":[0],
    "Partner":[1],
    "Dependents":[0],
    "Tenure":[12],
    "PhoneService":[1],
    "InternetService":[2],
    "MonthlyCharges":[75],
    "TotalCharges":[900],
})

# Align with dataset
for col in X.columns:
    if col not in new_customer.columns:
        new_customer[col] = 0

prediction = rf_model.predict(new_customer[X.columns])

print("\nChurn Prediction for New Customer:")
print("Churn" if prediction[0] == 1 else "Not Churn")
