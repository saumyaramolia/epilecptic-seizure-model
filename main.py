from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree, metrics
import pickle

scaler = StandardScaler()

data = pd.read_csv('Epileptic Seizure Recognition.csv')
data.head()
data.describe()
print(data.shape)
data.describe(include=object)
null_values = data.isnull().sum()
null_values.to_numpy()
data_1 = data.copy()
data_1.drop(['Unnamed', 'y'], axis=1, inplace=True)
data['y'].value_counts()
values = data['y'].value_counts()
plt.figure(figsize=(7, 7))
values.plot(kind='pie', fontsize=17, autopct='%.2f')
plt.legend(loc="best")
plt.show()
# fig, axs = plt.subplots(5, sharex=True, sharey=True)
# fig.set_size_inches(18, 24)
# labels = ["X15", "X30", "X45", "X60", "X75"]
# colors = ["r", "g", "b", 'y', "k"]
# fig.suptitle('Visual representation of different channels when stacked independently', fontsize=20)
# # loop over axes
# for i, ax in enumerate(axs):
#     axs[i].plot(data.iloc[:, 0], data[labels[i]], color=colors[i], label=labels[i])
#     axs[i].legend(loc="upper right")
#
# plt.xlabel('total number of observation', fontsize=20)
# plt.show()
data_2 = data.drop(["Unnamed"], axis=1).copy()
data_2["Output"] = data_2.y == 0
data_2["Output"] = data_2["Output"].astype(int)
data_2.y.value_counts()
data_2['y'] = data_2['y'].replace([2, 3, 4, 5], 0)
data_2.y.value_counts()
X = data_2.drop(['Output', 'y'], axis=1)
y = data_2['y']
counter = Counter(y)
# finding out the
print('Before', counter)
# oversampling the train dataset using SMOTE + ENN
smenn = SMOTEENN()
X_train1, y_train1 = smenn.fit_resample(X, y)

counter = Counter(y_train1)
print('After', counter)
X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.4, random_state=42)
# now we will be dividing it into further to get the validation set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# now we will going to scale the dataset
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("The shape of the training set is :{}".format(X_train.shape))
print("The shape of the testing set is :{}".format(X_test.shape))
print("The shape of the validation set is :{}".format(X_val.shape))

# now checking the accuracy on the decision tree classification


tree_eeg = tree.DecisionTreeClassifier()
tree = tree_eeg.fit(X_train, y_train)
# predicting
y_pred = tree.predict(X_val)
# Evaluating the model
precision = metrics.accuracy_score(y_pred, y_val) * 100
# print  the accuracy
print("Accuracy of the model by using the decision tree algorithm : {0:.2f}%".format(precision))

# calculate the FPR and TPR for all thresholds of the classification
y_pred = tree.predict(X_val)
# y_pred = y_pred[:, 1]
tree_fpr, tree_tpr, thresholds = metrics.roc_curve(y_val, y_pred)
tree_auc = metrics.roc_auc_score(y_val, y_pred)



# Save the trained model to a file
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(tree_eeg, file)
