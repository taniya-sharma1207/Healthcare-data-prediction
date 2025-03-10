import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
#df=dataframes -  > excel columns comma seperated value
df = pd.read_csv('Dummy_Healthcare_Dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Display information about the dataset
print(df.info())

# Display statistical summary of the dataset
print(df.describe())

# Display the column names of the dataset
print(df.columns)

# Display the count of missing values in each column
print(df.isnull().sum())

missing_values = df.isnull().sum()
print(missing_values)

# Check if there are any missing values in the dataset
any_missing = df.isnull().values.any()
print(any_missing)

# Handle missing data by dropping rows with missing values
#numerical = replace null vlaues with median of that column
#string= NaN
f'''
Replace numerical values with the median of their respective columns
for col in numerical_cols:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)

# Replace string values with NaN
for col in numerical_cols:
    median_value = df[col].median()
    df[col].fillna("NaN", inplace=True)#
'''

df = df.dropna()

# Select relevant columns for the analysis
df_selected = df[["Age", "Gender", "Admission Type", "Billing Amount"]]

# Encode categorical variables (Gender, Admission Type) using OneHotEncoder
f"""Encoding is the process of converting categorical data into a numerical format. 
OneHotEncoder achieves this by creating binary columns for each category. Each category is represented as a binary vector, where only one element is "1" (hot) and the rest are "0" (cold). 
This is particularly useful for algorithms that cannot work with categorical data directly.  
Example
For example, if we have a "Gender" column with values "Male" and "Female", 
OneHotEncoder will convert it into two columns: "Gender_Male" and "Gender_Female". 
If the original value is "Male", the encoded value will be [1, 0]. If the original value is "Female", the encoded value will be [0, 1]."""

encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_features = encoder.fit_transform(df_selected[["Gender", "Admission Type"]])
encoded_feature_names = encoder.get_feature_names_out(["Gender", "Admission Type"])

# Create a DataFrame with encoded values
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Concatenate the encoded categorical columns with the numerical columns
final_df = pd.concat([df_selected.drop(columns=["Gender", "Admission Type"]), encoded_df], axis=1)
# Train the multiple linear regression model
# Define independent (X) and dependent (y) variables
f"""final_df.drop(columns=["Billing Amount"]) means in the dataframe 'df' you are dropping Billing amount column just to keep gender and age columns as dependent variables."""
X = final_df.drop(columns=["Billing Amount"])
y = final_df["Billing Amount"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display model performance
print("R2 score:", r2)
print("RMSE:", rmse)

# Get model coefficients
coefficients = dict(zip(X.columns, model.coef_))

# Make predictions and evaluate the model
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Display model performance and coefficients
print("coefficients: ", coefficients)
print("r2: ", r2)
print("rmse", rmse)


# Scatter plot for actual vs predicted billing amount
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', alpha=0.5, label="Predicted vs Actual")
plt.plot(y, y, color='red', linestyle="dashed", label="Perfect Fit (y=x)")

plt.xlabel("Actual Billing Amount")
plt.ylabel("Predicted Billing Amount")
plt.title("Actual vs Predicted Billing Amount")
plt.legend()
plt.grid(True)
plt.show()