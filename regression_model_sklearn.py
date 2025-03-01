import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing #Example dataset
import os
import pickle

def get_regression_data(test_size=0.2):
    # 1. Load Data (Using California Housing dataset as an example)
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop('MedHouseVal', axis=1)  # Features
    y = df['MedHouseVal']  # Target (median house value)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Feature Scaling (Important for some regression models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


if __name__ == "__main__":

    X_train_scaled, y_train, X_test_scaled, y_test = get_regression_data()

    # 4. Choose and Train the Model (Linear Regression)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 5. Make Predictions
    y_pred = model.predict(X_test_scaled)

    # 6. Evaluate the Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    #Example of predicting a single house price.
    new_house = X_test_scaled[0].reshape(1, -1) #Gets the first row of scaled test data.
    predicted_price = model.predict(new_house)
    print(f"Predicted price for the first test house: {predicted_price[0]}")

    #Example of predicting all test set prices.
    predicted_prices = model.predict(X_test_scaled)
    print(f"Predicted prices for test set first 5: {predicted_prices[:5]}")

    model_name = "housing_regression.pkl"
    pth = os.path.join(os.getcwd(), 'model_dump', model_name)

    pickle.dump(model, open(pth,'wb'))
    print(f"{model_name} save {pth} successfully")