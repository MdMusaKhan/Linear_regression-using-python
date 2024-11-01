import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_excel(r"E:\centennial college\4th Semester\AI for Software Developers\real+estate+valuation+data+set\Real estate valuation data set.xlsx")

# Display the first few rows of the dataset
print("Dataset Head:")
print(data.head())

#display columns:

print("Columns in the dataset:")
print(data.columns)


# Preprocessing: Select features and target variable
# Using relevant columns based on the dataset structure
data['X1 transaction date'] = pd.to_datetime(data['X1 transaction date']).dt.year  # Convert date to year

features = data[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X6 longitude']]
target = data['Y house price of unit area']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation:\nMean Squared Error: {mse}\nR² Score: {r2}")

# Function to predict house price based on user input
def predict_house_price():
    print("\nEnter the values for the following attributes:")
    transaction_year = int(input("Transaction year (e.g., 2012): "))
    house_age = float(input("House age: "))
    distance_to_mrt = float(input("Distance to the nearest MRT (in meters): "))
    num_convenience_stores = int(input("Number of convenience stores: "))
    longitude = float(input("Longitude: "))
    
    input_data = np.array([[transaction_year, house_age, distance_to_mrt, num_convenience_stores, longitude]])
    predicted_price = model.predict(input_data)
    
    print(f"\nPredicted house price per unit area: {predicted_price[0]:.2f}")

# Run the prediction function
predict_house_price()
