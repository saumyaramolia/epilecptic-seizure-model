import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle

# Read data
data = pd.read_csv('Epileptic Seizure Recognition.csv')

# Handle missing values (if any)
null_values = data.isnull().sum()

# Drop rows with missing values
data.dropna(inplace=True)

# Split features and target
X = data.drop(['Unnamed', 'y'], axis=1)
y = data['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree classifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train_scaled, y_train)

# Evaluate Decision Tree model
y_pred = tree_clf.predict(X_test_scaled)
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("Accuracy of the model using Decision Tree algorithm: {:.2f}%".format(accuracy))

# Save trained model to a file
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(tree_clf, file)
