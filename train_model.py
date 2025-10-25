import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = pd.read_csv('loans.csv')  # Updated to use loans.csv
data = data.ffill()
feature_cols = ['ApplicantIncome', 'LoanAmount', 'Credit_History']

for col in ['Education', 'Married', 'Property_Area']:
    if col in data.columns:
        feature_cols.append(col)


# Encode categorical features
for col in ['Education', 'Married', 'Property_Area']:
    if col in data.columns:
        data[col] = data[col].astype('category').cat.codes


# Prepare model input as NumPy arrays
X = data[feature_cols].to_numpy()
y = data['Loan_Status'].map({'Y': 1, 'N': 0}).to_numpy()


# Statistical analysis with NumPy
print("\nStatistical Analysis (NumPy):")
for i, col in enumerate(feature_cols):
    if np.issubdtype(data[col].dtype, np.number):
        arr = data[col].to_numpy()
        print(f"{col} - mean: {np.mean(arr):.4f}, variance: {np.var(arr):.4f}")

# Correlation matrix for numeric features
numeric_cols = [col for col in feature_cols if np.issubdtype(data[col].dtype, np.number)]
if len(numeric_cols) > 1:
    corr_matrix = np.corrcoef(data[numeric_cols].to_numpy(), rowvar=False)
    print("\nCorrelation matrix:")
    print(pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter tuning with GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Removed use_label_encoder parameter as it is deprecated
xgb_clf = xgb.XGBClassifier(eval_metric='logloss')
grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Print feature importances
importances = model.feature_importances_
print("Feature importances:")
for name, importance in zip(feature_cols, importances):
    print(f"{name}: {importance:.4f}")


# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# For AUC, need predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print(f'Model Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print(f'AUC: {auc:.2f}')


# Model explainability with SHAP
try:
    import shap
    print("Generating SHAP summary plot...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
except ImportError:
    print("SHAP is not installed. Run 'pip install shap' to enable model explainability plots.")

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
print('Model saved to model.pkl')
