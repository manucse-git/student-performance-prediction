import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'study_hours': [2, 4, 6, 8, 10],
    'attendance': [60, 70, 75, 85, 90],
    'marks': [50, 55, 65, 75, 85]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['study_hours', 'attendance']]
y = df['marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)

print("Predicted Marks:", predictions)
print("Mean Squared Error:", mse)
