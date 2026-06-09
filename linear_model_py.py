"Linear model.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [20, 40, 50, 60, 80]
}

df = pd.DataFrame(data)
X = df[['Hours']]
y = df['Marks']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


model = LinearRegression()


model.fit(X_train, y_train)


pred = model.predict(X_test)


mse = mean_squared_error(y_test, pred)

# Output
print("Predicted Marks:", pred)
print("Mean Squared Error:", mse)

# Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, pred, color='red',
         linewidth=2, label='Predicted Line')


plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Linear Regression Model")
plt.legend()
plt.show()

