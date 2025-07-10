from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the iris dataset
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)  # Binary classification: class 0 vs not class 0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Class 0', 'Class 0'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# classification matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))