import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Dummy_Healthcare_Dataset.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Convert 'Test Results' into categorical classes
f""" 
0: abnormal 
1: normal
2: Inconclusive
"""
df['Test Results'] = df['Test Results'].apply(lambda x: 2 if x == 'Inconclusive' else (1 if x == 'Normal' else 0))

# Select relevant features
features = ["Age", "Gender", "Admission Type", "Medical Condition", "Insurance Provider", "Medication"]
X = df[features]
y = df["Test Results"]

# Encode categorical variables
categorical_features = ["Gender", "Admission Type", "Medical Condition", "Insurance Provider", "Medication"]
numeric_features = ["Age"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create a pipeline with preprocessing and decision tree classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Display classification report
print(classification_report(y_test, y_pred, target_names=['Abnormal', 'Normal', 'Inconclusive']))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Analyze misclassifications
misclassified_indices = np.where(y_test != y_pred)[0]
misclassified_samples = X_test.iloc[misclassified_indices]
print("Misclassified Samples:")
print(misclassified_samples)

# Provide insights and recommendations
print("Insights and Recommendations:")
print("1. The model's performance can be improved by collecting more data and using more advanced models.")
print("2. Misclassifications can be analyzed to understand patterns and improve data quality.")
print("3. Healthcare professionals should consider additional tests for patients with inconclusive results.")
print("4. Regular model retraining with updated data can help maintain accuracy.")

# Plot confusion matrix (OPTIONAL)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Abnormal', 'Normal', 'Inconclusive'], yticklabels=['Abnormal', 'Normal', 'Inconclusive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()