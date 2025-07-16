# Task 1 - Perform Feature Engineering
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# print("Original DataFrame:")
# print(df.head())

# select relevant features
df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Survived']]

# Handle missing values
df.fillna({
    'Age': df['Age'].median(),
    'Embarked': df['Embarked'].mode()[0]
}, inplace=True)

# define features and target
X = df.drop(columns='Survived')
y = df['Survived']

# apply feature scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Fare']),
        ('cat', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked'])
    ]
)

X_preprocssed = preprocessor.fit_transform(X)

# train and evaluate logistic regression model
log_model = LogisticRegression()
log_scores = cross_val_score(log_model, X_preprocssed, y, cv=5, scoring='accuracy')
print(f"Logistic Regression Cross-Validation Scores: {log_scores.mean():.2f}")

# Train and evaluate random forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf_model, X_preprocssed, y, cv=5, scoring='accuracy')
print(f"Random Forest Accuracy {rf_scores.mean():.2f}")

# Define the hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform the grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_preprocssed, y)

# display the best parameters and score
print("Best Parameters for Random Forest:", grid_search.best_params_)
print("Best Cross-Validation Score for Random Forest:", grid_search.best_score_)
