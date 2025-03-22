import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model, metrics, datasets
import matplotlib.pyplot as plt


# Load the Digits dataset
dataset = datasets.load_digits(as_frame=True)
# Process and split data to train and test sets
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.25, random_state=123
)

# Print Values from Dataset
print(f"x = {x.shape}")
print(f"y = {y.shape}")
print(y[:50])

# Train logistic regression
regr = linear_model.LogisticRegression(solver="lbfgs", max_iter=10000, random_state=123)
regr.fit(x_train, y_train)

# Calculate predicted values
y_pred = regr.predict(x_test)

# Calculate performance metrics: Accuracy, Precision, Recall and F1
prec = metrics.precision_score(y_true=y_test, y_pred=y_pred, average="macro")
reca = metrics.recall_score(y_true=y_test, y_pred=y_pred, average="macro")
f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred, average="macro")

# Print values
print("prec", prec)
print("reca", reca)
print("f1", f1)

row = 50

first_row = x.iloc[row].values
image = first_row.reshape(8, 8)

plt.imshow(image, cmap="gray")
plt.show()
