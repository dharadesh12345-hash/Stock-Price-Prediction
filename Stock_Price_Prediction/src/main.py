import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("data/archive (2)/HistoricalQuotes.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Remove '$' symbol and convert price columns into numbers
price_columns = ["Close/Last", "Open", "High", "Low"]

for column in price_columns:
    data[column] = data[column].str.replace("$", "", regex=False)
    data[column] = data[column].astype(float)

# Convert Date column into date format
data["Date"] = pd.to_datetime(data["Date"])

# Display first 5 rows
print(data.head())

# Create features and target
X = data[["Open", "High", "Low", "Volume"]]
Y = data["Close/Last"]

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train model
model = LinearRegression()
joblib.dump(model, "model/stock_model.pkl")
print("Model saved successfully!")

# Make predictions
predictions = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(predictions[:10])

print("\nActual Values:")
print(Y_test.head(10).values)

# Evaluate model
mae = mean_absolute_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

print("\nModel Performance")
print("MAE:", mae)
print("R2 Score:", r2)

# User input for custom prediction
print("\nEnter values for custom stock prediction")

open_price = float(input("Enter Open Price: "))
high_price = float(input("Enter High Price: "))
low_price = float(input("Enter Low Price: "))
volume = int(input("Enter Volume: "))

new_data = [[open_price, high_price, low_price, volume]]

predicted_price = model.predict(new_data)

print("Predicted Closing Price:", predicted_price[0])

# Actual vs Predicted graph
plt.figure(figsize=(10, 5))

plt.plot(Y_test.values[:100], label="Actual Price")
plt.plot(predictions[:100], "--", label="Predicted Price")

plt.xlabel("Test Data Points")
plt.ylabel("Stock Price")
plt.title("Actual vs Predicted Stock Prices")

plt.legend()

plt.show()