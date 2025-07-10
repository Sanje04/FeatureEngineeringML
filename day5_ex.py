# load bike sharing daily csv
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/parthvr/bike-sharing/refs/heads/main/bike_sharing_daily.csv')

# # Display the first 5 rows of the dataset
# print("First 5 Rows of the Dataset:")
# print(df.head())
# # Display dataset information
# print("\nDataset Information:") 
# print(df.info())

# convert 'dteday' to datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# create new features
df['day_of_week'] = df['dteday'].dt.day_name()
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

# print("\nNew Features Created:")
# print(df[['dteday', 'day_of_week', 'month', 'year']].head())

# Select feature and target
X = df[['temp']]
y = df['cnt']

# Apply polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)


# # Display the transformed features
# print("\nTransformed Features:")
# print(pd.DataFrame(X_poly, columns=['temp', 'temp^2']).head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train and evaluate the model with the original feature
model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
mse_original = mean_squared_error(y_test, y_pred_original)

# Train and evaluate the model with the polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_pred_poly = model_poly.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# compare the results
print("\nMean Squared Error with Original Feature:", mse_original)
print("Mean Squared Error with Polynomial Features:", mse_poly)
