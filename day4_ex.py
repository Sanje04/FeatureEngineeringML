from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# Load the diabetes dataset
# This dataset contains information about diabetes patients, including various health metrics and a target variable indicating disease
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
# this is the column that we want to predict
df['target'] = data.target

# # Display the first 5 rows of the dataset
# print(df.head())
# # Display dataset information
# print("\nDataset Information:", df.info())

# Calculate the correlation matrix
correlation_matrix = df.corr()

# plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix of Diabetes Dataset')
# plt.show()

# select features with high correlation with the target variable
correlated_features = correlation_matrix['target'].sort_values(ascending=False)
# print("\nFeatures correlated with target variable:")
# print(correlated_features)

# seperate feature and target variable
X = df.drop(columns=['target'])
y = df['target']

# Calculate mutual information between features and target variable
mi_scores = mutual_info_regression(X, y)

# Create a DataFrame to hold the feature names and their mutual information scores
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

print("\nMutual Information Scores:")
print(mi_df)

from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Train a random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances from Random Forest:")
print(importance_df)

# plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.title('Feature Importances from Random Forest')
plt.show()