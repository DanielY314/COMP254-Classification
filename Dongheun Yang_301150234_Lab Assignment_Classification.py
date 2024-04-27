"""
Created on Tue Feb 10 01:16:26 2024

@author: USER
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the MNIST data
mnist = fetch_openml('mnist_784')

# Load the data into a pandas dataframe named MNIST_dongheun
MNIST_dongheun = pd.DataFrame(mnist.data)

# List the keys
keys = mnist.keys()

# Assign the data to a ndarray named X_dongheun
X_dongheun = mnist.data.to_numpy()

# Assign the target to a variable named y_dongheun
y_dongheun = mnist.target.to_numpy()

# Print the types of X_dongheun and y_dongheun
types_X_y = (X_dongheun.dtype, y_dongheun.dtype)
print("types of X and y: ", types_X_y)

# Print the shape of X_dongheun and y_dongheun
shapes_X_y = (X_dongheun.shape, y_dongheun.shape)
print("shapes for X and y: ", shapes_X_y)

# Store values in variables some_digit1, some_digit2, some_digit3
some_digit1 = X_dongheun[7]
some_digit2 = X_dongheun[5]
some_digit3 = X_dongheun[0]

# Reshape the digits
some_digit1_image = some_digit1.reshape(28, 28)
some_digit2_image = some_digit2.reshape(28, 28)
some_digit3_image = some_digit3.reshape(28, 28)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(some_digit1_image, cmap='binary')
axes[0].set_title('Digit 1')
axes[0].axis('off')

axes[1].imshow(some_digit2_image, cmap='binary')
axes[1].set_title('Digit 2')
axes[1].axis('off')

axes[2].imshow(some_digit3_image, cmap='binary')
axes[2].set_title('Digit 3')
axes[2].axis('off')

plt.show()

# Assuming y_dongheun is already defined as the target variable from the MNIST dataset
# Step 9: Change the type of y to uint8
y_dongheun = y_dongheun.astype(np.uint8)

# Step 10: Transform the target variable to 3 classes
y_transformed = np.where(y_dongheun <= 3, 0, np.where(y_dongheun <= 6, 1, 9))

# Step 11: Print the frequencies of each of the three target classes
unique, counts = np.unique(y_transformed, return_counts=True)
class_frequencies = dict(zip(unique, counts))
print("class_frequencies: " ,class_frequencies)

# Plotting the bar chart for the frequencies
plt.figure(figsize=(8, 6))
plt.bar(class_frequencies.keys(), class_frequencies.values(), color=['blue', 'orange', 'green'])
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Frequency of Each Class After Transformation')
plt.xticks([0, 1, 9])
plt.show()

# Step 12: Split your data into train, test without using sklearn's train_test_split
X_train, X_test = X_dongheun[:50000], X_dongheun[-20000:]
y_train, y_test = y_transformed[:50000], y_transformed[-20000:]

# Output shapes just to confirm splitting correctly, not part of the requirements
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Train a Naive Bayes classifier using the training data
NB_clf_dongheun = GaussianNB()
NB_clf_dongheun.fit(X_train, y_train)

# Use 3-fold cross validation to validate the training process
cv_scores = cross_val_score(NB_clf_dongheun, X_train, y_train, cv=3)
print("cv_scores: ", cv_scores)

# Use the model to score the accuracy against the test data
y_pred_test = NB_clf_dongheun.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("accuracy test: " ,accuracy_test)


# Generate the accuracy matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)

some_digit_pred = NB_clf_dongheun.predict(X_test[:3])

# Actual class labels for the three selected digits
actual_labels = y_transformed[[7, 5, 0]]

print("predicted three variables: ", some_digit_pred)
print("Actual labels for the three variables: ", actual_labels)

# Train a Logistic Regression classifier using the "lbfgs" solver
LR_clf_dongheun_lbfgs = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1200, tol=0.1)
LR_clf_dongheun_lbfgs.fit(X_train, y_train)


# Train a Logistic Regression classifier using the "saga" solver
LR_clf_dongheun_saga = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1200, tol=0.1)
LR_clf_dongheun_saga.fit(X_train, y_train)

# Use 3-fold cross validation on the training data for both models
cv_scores_lbfgs = cross_val_score(LR_clf_dongheun_lbfgs, X_train, y_train, cv=3)
cv_scores_saga = cross_val_score(LR_clf_dongheun_saga, X_train, y_train, cv=3)

y_pred_test_lbfgs = LR_clf_dongheun_lbfgs.predict(X_test)
accuracy_test_lbfgs = accuracy_score(y_test, y_pred_test_lbfgs)

y_pred_test_saga = LR_clf_dongheun_saga.predict(X_test)
accuracy_test_saga = accuracy_score(y_test, y_pred_test_saga)

conf_matrix_lbfgs = confusion_matrix(y_test, y_pred_test_lbfgs)
conf_matrix_saga = confusion_matrix(y_test, y_pred_test_saga)

precision_lbfgs = precision_score(y_test, y_pred_test_lbfgs, average='weighted')
precision_saga = precision_score(y_test, y_pred_test_saga, average='weighted')

recall_lbfgs = recall_score(y_test, y_pred_test_lbfgs, average='weighted')
recall_saga = recall_score(y_test, y_pred_test_saga, average='weighted')
# Correctly print cross-validation scores
print("LBFGS Cross-Validation Scores:", cv_scores_lbfgs)
print("SAGA Cross-Validation Scores:", cv_scores_saga)

# Print accuracy, precision, and recall
print("LBFGS Test Accuracy:", accuracy_test_lbfgs)
print("SAGA Test Accuracy:", accuracy_test_saga)
print("LBFGS Precision:", precision_lbfgs)
print("LBFGS Recall:", recall_lbfgs)
print("SAGA Precision:", precision_saga)
print("SAGA Recall:", recall_saga)


# Visualization for LBFGS Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_lbfgs, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix for LBFGS Solver')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Repeat for SAGA
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_saga, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix for SAGA Solver')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()




some_digit_pred_lr = LR_clf_dongheun_lbfgs.predict(X_test[:3])
print("predicted some digit: ", some_digit_pred)










