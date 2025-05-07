from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
from scipy import stats
# "C:/Users/KIIT/Minor Project/Mini_Project(Preterm_birth_detection)/FrontEnd+Backend_part/Dataset/synthetic_preterm_3000_final_95.csv"
# Load the dataset
file_path = "C:/Users/KIIT/Minor Project/Mini_Project(Preterm_birth_detection)/FrontEnd+Backend_part/Dataset/synthetic_preterm_3000_final_95.csv"
df = pd.read_csv(file_path)

# Drop low-impact features
df_cleaned = df.drop(columns=["STD", "lenght of contraction"], errors='ignore')

# Handle outliers using Z-score
z_scores = np.abs(stats.zscore(df_cleaned.select_dtypes(include=["number"])))
df_cleaned = df_cleaned[(z_scores < 3).all(axis=1)]

# Remove outliers using IQR for specific column
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

df_cleaned = remove_outliers_iqr(df_cleaned, 'Risk Factor Score')

# Feature Engineering
df_cleaned['Contraction_Risk_Interaction'] = df_cleaned['Count Contraction'] * df_cleaned['Risk Factor Score']

# Define features and target variable
X = df_cleaned.drop(columns=['Pre-term'], errors='ignore')
y = df_cleaned['Pre-term']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost Model with Regularization
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=5, 
                              learning_rate=0.1, reg_lambda=1, reg_alpha=1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_model.save_model("xgboost_model.json")  # ✅ Save in the new format

# Train other models
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

et_model = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
et_model.fit(X_train, y_train)

cat_model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, verbose=0, random_state=42)
cat_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)

# Evaluate models
models = {"XGBoost": xgb_model, "Extra Trees": et_model, "CatBoost": cat_model, "SVM": svm_model, "Logistic Regression": lr_model}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print("Actual vs Predicted:")
    result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    print(result_df.head(10))  # Print first 10 rows
    print("="*60)

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("="*60)

# Select best model based on accuracy
best_model = max(models, key=lambda m: accuracy_score(y_test, models[m].predict(X_test)))
print(best_model)
# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")
joblib.dump(models[best_model], "best_model.pkl")
print(f"Best model ({best_model}) saved as best_model.pkl")
