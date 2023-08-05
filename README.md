# Project-python
 Predict a student's percentage based on their study hours using linear regression. The model estimates the percentage score a student may achieve given the number of hours they study.
 
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing Datasets
df = pd.read_csv(r"C:\Users\Shrawani\Downloads\scores.csv")
df.head()
df

# Visualizing Data
df.Scores.plot()

# Finding Correlation
df.corr()

# Importing Library
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Create and fit the linear regression model
X = df[['Hours']]
y = df['Scores']
lr.fit(X, y)

# Getting Coefficient and Intercept

slope = lr.coef_[0]
intercept = lr.intercept_
print("Slope (Coefficient):", slope)
print("Intercept:", intercept)

df['Predicted_Scores'] = lr.predict(X)

# Ploting Bestfit Line
plt.scatter(df['Hours'], df['Scores'], label='Actual Scores')
plt.plot(df['Hours'], df['Predicted_Scores'], color='red', label='Regression Line')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.legend()
plt.show()

# Making Predictions using Model
hours_to_predict = float(input("Enter a time: ")) 
predicted_score = lr.predict([[hours_to_predict]])
print(f"Predicted Score for {hours_to_predict} study hours: {predicted_score[0]}")
