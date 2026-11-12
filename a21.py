
# Project A21: Airline Passenger Growth

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# STEP 1: Load the dataset
df = pd.read_csv("AirPassengers.csv")

df = df.rename(columns={"#Passengers": "passengers"})

print("Rows and columns:", df.shape)

# STEP 2: Clean the data
df["t"] = range(len(df))

df = df.dropna(subset=["t", "passengers"])

# STEP 3: Choose input X and output y
X = df[["t"]]
y = df["passengers"]

# STEP 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 6: Test the model
y_pred = model.predict(X_test)

print("R2 score:", round(r2_score(y_test, y_pred), 3))

print("Average error:",
      round(mean_absolute_error(y_test, y_pred), 2),
      "thousand")

print("Slope m:", round(model.coef_[0], 4))

print("Intercept c:", round(model.intercept_, 2))

# STEP 7: Predict future passengers
new = pd.DataFrame({"t": [144]})

print("Predicted passengers for Jan 1961:",
      round(model.predict(new)[0], 2), "thousand")

# Draw the graph
line_x = pd.DataFrame({"t": sorted(X["t"])})

plt.scatter(
    X_test["t"], y_test,
    s=12, alpha=0.6,
    label="Actual (test data)"
)

plt.plot(
    line_x["t"],
    model.predict(line_x),
    color="red",
    label="Regression line"
)

plt.xlabel("Month number (0 = Jan 1949)")
plt.ylabel("Passengers (thousands)")

plt.title("Airline Passenger Growth")

plt.legend()
plt.show()
