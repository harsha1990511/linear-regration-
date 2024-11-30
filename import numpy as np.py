import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
np.random.seed(42)  # For reproducibility 
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature 
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with noise 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
model = LinearRegression() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
print("Model Coefficients:", model.coef_) 
print("Model Intercept:", model.intercept_) 
print("Mean Squared Error:", mse) 
print("R^2 Score:", r2) 
Linear regression 
plt.figure(figsize=(10, 6)) 
plt.scatter(X, y, color='blue', label='Data Points') 
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Random Dataset') 
plt.xlabel('X') 
plt.ylabel('y') 
plt.legend() 
plt.show()