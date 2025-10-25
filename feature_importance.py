import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('loans.csv')

# Drop rows with missing target
df = df.dropna(subset=['Loan_Status'])

# Prepare features and target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Encode categorical variables
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Missing')
    X[col] = LabelEncoder().fit_transform(X[col])

# Encode target
y = LabelEncoder().fit_transform(y)

# Fill missing values for numeric columns
X = X.fillna(X.median())

# Fit RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Print feature importances in descending order
importances = model.feature_importances_
features_sorted = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)

print("Feature Importances (descending):")
for feature, importance in features_sorted:
    print(f"{feature}: {importance:.4f}")
    'ApplicantIncome', 'LoanAmount', 'Credit_History',
    'Education', 'Married', 'Property_Area', 'Gender',
    'Dependents', 'Self_Employed', 'CoapplicantIncome', 'Loan_Amount_Term'





