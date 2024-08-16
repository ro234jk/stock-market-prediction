import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("prices.csv")

# Display the first few rows to confirm it's loaded correctly
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# Select the 'Close' price as the feature and the target for prediction
forecast_col = 'Close'
forecast_out = 5  # Predicting 5 days into the future

# Create a new column 'Label' with shifted 'Close' prices
df['Label'] = df[forecast_col].shift(-forecast_out)

# Prepare the feature and target arrays
X = df[[forecast_col]].values
X = scale(X)  # Standardize the features
X_lately = X[-forecast_out:]  # Data for which we'll make future predictions
X = X[:-forecast_out]  # Data for training and testing

df.dropna(inplace=True)
y = df['Label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model with the training data
model.fit(X_train, y_train)

# Check the accuracy of the model on the test data
score = model.score(X_test, y_test)
print(f"Model Test Accuracy: {score:.2f}")
# Make predictions for the next 5 days
forecast = model.predict(X_lately)

# Display the predictions
print("Predicted stock prices for the next 5 days:")
print(forecast)

