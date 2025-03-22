import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model, metrics

# Load the 4G_Speed.csv dataset
df = pd.read_csv("4G_Speed.csv")
df_clean = df[["DL_bitrate", "UL_bitrate"]]
print(df_clean.head(n=5))
# Keep only DL_bitrate and UL_bitrate columns,
# drop any rows where UL_bitrate is 0
x = df_clean.drop("UL_bitrate", axis=1)
y = df_clean["UL_bitrate"]


# Plot the dataset, DL_bitrate on x axis

sns.lmplot(x="DL_bitrate", y="UL_bitrate", data=df_clean)
plt.show()

# Process and split data to train and test sets

x = np.array(df_clean["DL_bitrate"]).reshape(-1, 1)
y = np.array(df_clean["UL_bitrate"]).reshape(-1, 1)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.25, random_state=123
)

# Perform linear regression
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)
# Draw scatter plots of results, blue for test data,
# green for predicted y based on x_test

for col in x_test.columns:
    plt.scatter(x_test[col], y_test, color="blue")
    plt.scatter(x_test[col], y_pred, color="green")
    plt.show("Test")

# Calculate performance metrics: MSE, RMSE, MAE, MAPE and R2
mse = metrics.mean_squared_error(y_true=y_test, y_pred=y_pred, squared=True)

print("MSE: ", mse)
