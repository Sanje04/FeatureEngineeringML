import pandas as pd

# load the titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# # display dataset information
# print("Dataset Information:")
# print(df.info())

# # display the first 5 rows of the dataset
# print("\nFirst 5 Rows of the Dataset:")
# print(df.head())

# Seperate features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# print("\nCategorical Features:")
# print(categorical_features)
# print("\nNumerical Features:")
# print(numerical_features)

# Display summary of categorical features
print("\nSummary of Categorical Features:")
for feature in categorical_features:
    print(f"\n{feature}:")
    print(df[feature].value_counts(dropna=False))
    
    
# Display summary of numerical features
print("\nSummary of Numerical Features:")
print(df[numerical_features].describe())