# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso # L1 regularization

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('regularization_data.csv', header = None)
X = train_data.iloc[:,:-1] # From the first to the 2nd last column
y = train_data.iloc[:,-1] # The last column

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)