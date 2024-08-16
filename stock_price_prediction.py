import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out)
    X = np.array(df[[forecast_col]])
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    label.dropna(inplace=True)
    y = np.array(label)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, Y_train, Y_test, X_lately

# Load data
df = pd.read_csv("prices.csv")
df = df[df.symbol == "GOOG"]

# Define variables
forecast_col = 'close'
forecast_out = 5
test_size = 0.2

# Prepare data
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)

# Train model
learner = LinearRegression()
learner.fit(X_train, Y_train)

# Make predictions
score = learner.score(X_test, Y_test)
forecast = learner.predict(X_lately)

# Output results
response = {
    'test_score': score,
    'forecast_set': forecast
}

print(response)
