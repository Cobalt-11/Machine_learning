import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model, metrics, datasets

# Load the Breast Cancer dataset
dataset = datasets.load_breast_cancer(as_frame=True)

# Processing data
x = dataset.data
y = dataset.target

# Spliting data to train and test

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.25, random_state=123
)


# Train logistic regression

regr = linear_model.LogisticRegression(
    solver="liblinear", max_iter=100, random_state=123
)
regr.fit(x_train, y_train)


# Calculate predicted values
y_pred = regr.predict(x_test)

# Calculate performance metrics: Accuracy, Precision, Recall and F1

accu = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
prec = metrics.precision_score(y_true=y_test, y_pred=y_pred)
reca = metrics.recall_score(y_true=y_test, y_pred=y_pred)
f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred)

# Showcasing Performance metrics
print("Accuracy: ", accu)
print("Precision: ", prec)
print("Recall: ", reca)
print("f1_score: ", f1)
