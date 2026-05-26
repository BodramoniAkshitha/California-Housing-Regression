# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()

# Convert to DataFrame
data = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add target column
data["Price"] = housing.target

# Display first 5 rows
print(data.head())

# Features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Calculate error
mse = mean_squared_error(y_test, predictions)

print("\nModel Trained Successfully!")
print("Mean Squared Error:", mse)

# Plot Actual vs Predicted
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()