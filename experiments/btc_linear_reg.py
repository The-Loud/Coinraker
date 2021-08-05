"""
This module runs a basic linear regression on log-transformed and normalized data.
It serves as a baseline to see if the accuracy of the MC DCNN is better.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load up the data
data = pd.read_csv("./data/training.csv")
X = data.drop(["usd", "load_date", "crypto"], axis=1)
y = np.log(data["usd"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69
)

# Preprocess
si = SimpleImputer(missing_values=np.nan, strategy="mean")
si.fit(X_train)

X_train = si.transform(X_train)
X_test = si.transform(X_test)

# Scale the data
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

# Do the thing
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(lr.score(X_test, y_test))

plt.figure()
ax = plt.axes()
plt.scatter(X_test[:, 0], y_test)
ax.scatter(X_test[:, 0], y_pred)
plt.show()


# about 90 is the score to beat.
print(f"Mean Square Error: {mean_squared_error(y_pred, y_test)}")
