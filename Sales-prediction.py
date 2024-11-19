# I have dowmloaded advertising.csv from kaggle dataset for sales data from various advertising spends for sales prediction from different advertisement channels. 
# Dataset link - https://www.kaggle.com/datasets/brsahan/advertising-spend-vs-sales/code

# import the neccessary modules for linear regression module

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load the data into dataframe and visulaize the data

dataset = pd.read_csv("Advertising.csv")
dataset.info()
dataset.head()
dataset.tail()
sns.barplot(x="radio",y="sales",data = dataset) #you can use pairplot for visulaizing all the columns data
plt.show()

sns.heatmap(dataset.corr(),annot = True,cmap = 'summer')
plt.show()

# make the train data , test data and ready for model training

x = dataset[['TV','radio','newspaper']]
y = dataset['sales']

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2 , random_state =40)

print(f"Training set size: {x_train.shape}")
print(f"Testing set size: {x_test.shape}")

# train the mode and see the coefficient and intercept

model = LinearRegression()
model.fit(x_train,y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# predict the sales

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2 Score):", r2)

#visualizing the prediction
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()

# output analysis 

variance = np.var(y_test)
std_dev = np.std(y_test)
print("Variance of Sales:", variance)
print("Standard Deviation of Sales:", std_dev)

# our ouput MSE is significantly smaller than variance so our model is performing better 

#as next steps we can consider feature engineering and feature scaling to improve the model performance
