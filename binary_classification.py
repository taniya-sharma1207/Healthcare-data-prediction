import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Dummy_Healthcare_Dataset.csv')

# Handle missing values
df = df.dropna()

# Convert 'Test Results' into binary classes
df['Test Results'] = df['Test Results'].apply(lambda x: 1 if x == 'Normal' else 0)

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

# Create a pipeline with preprocessing and logistic regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Display classification report
print(classification_report(y_test, y_pred))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Abnormal/Inconclusive', 'Normal'], yticklabels=['Abnormal/Inconclusive', 'Normal'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Identify key predictors
model = pipeline.named_steps['classifier']
feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features).tolist() + numeric_features
coefficients = model.coef_[0]
key_predictors = dict(zip(feature_names, coefficients))
print("Key Predictors:")
print(key_predictors)

# Explore potential reasons for misclassifications
misclassified_indices = np.where(y_test != y_pred)[0]
misclassified_samples = X_test.iloc[misclassified_indices]
print("Misclassified Samples:")
print(misclassified_samples)