import pandas as pd
# load the titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Display dataset information
print("Dataset Information:")
print(df.info())

# Display the first 5 rows of the dataset
print("\nFirst 5 Rows of the Dataset:")
print(df.head())


# Apply one-hot encoding to categorical features
df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# display the encoded dataset
print("\nDataset after One-Hot Encoding:")
print(df_one_hot.head())
# it creates new columns and drops some of them

# Apply label encoding
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# display encoded dataset
print("\nDataset after Label Encoding:")
print(df[['Pclass', 'Pclass_encoded']].head())

# apply frequency encoding
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# display frequency encoded feature
print("\nDataset after Frequency Encoding:")
print(df[['Ticket', 'Ticket_frequency']].head())

df_one_hot.dropna(inplace=True)

X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = df_one_hot['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the Logistic Regression model:", accuracy)