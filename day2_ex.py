from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Load the iris dataset
data = load_iris()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display the first 5 rows of the dataset
print("Dataset Info:")
print(x.describe())
print("\n Target Classses:", data.target_names)

# Train the KNN without scaling
# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# predict and evaluate the model
y_pred = knn.predict(x_test)
print("\nAccuracy without scaling:", accuracy_score(y_test, y_pred))

# Apply Min-Max scaling
# Created a MinMaxScaler instance
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(x)

# Split the scaled dataset into training and testing sets
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the KNN classifier on the scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(x_train_scaled, y_train_scaled)

# Predict and evaluate the model on the scaled data
y_pred_scaled = knn_scaled.predict(x_test_scaled)
print("\nAccuracy with Min-Max scaling:", accuracy_score(y_test_scaled, y_pred_scaled))

# Apply Standard scaling
# Created a StandardScaler instance
scaler_standard = StandardScaler()
x_standard_scaled = scaler_standard.fit_transform(x)

# split the scaled dataset into training and testing sets
x_train_standard_scaled, x_test_standard_scaled, y_train_standard_scaled, y_test_standard_scaled = train_test_split(x_standard_scaled, y, test_size=0.2, random_state=42)

# Train the KNN classifier on the standard scaled data
knn_standard_scaled = KNeighborsClassifier(n_neighbors=5)
knn_standard_scaled.fit(x_train_standard_scaled, y_train_standard_scaled)

# Predict and evaluate the model on the standard scaled data
y_pred_standard_scaled = knn_standard_scaled.predict(x_test_standard_scaled)
print("\nAccuracy with Standard scaling:", accuracy_score(y_test_standard_scaled, y_pred_standard_scaled))
# Display the accuracy results
