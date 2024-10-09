from visualization import graph
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
import numpy as np
import pandas as pd 

#Perform the Linear Regression Model
def linearRegression(df):

    x = df[['Usage','Average Temperature']] #Independent Variables
    y = df['Charge'] #Dependent Variable

    model = LinearRegression() #Create Object of Linear Regression Class
    model.fit(x,y) #Train the model

    return model

#Predict Values based on the training model
def predict(model, data):

    x = data[['Usage', 'Average Temperature']]

    predictions = model.predict(x) #Predict Y values based on X values

    print("Predictions:")
    print(predictions)

    return predictions

#Get the statistical information 
def additional_info(data, predictions):

    #Get the list of Residuals and Max Residual and print them
    residuals = []
    max_residual = float('-inf')
    print("Residuals: ")
    for idx in range(len(predictions)):
        residual = data['Charge'][idx] - predictions[idx]
        residuals.append(residual) #Append the subtraction of the actual value and the predicted value
        print(f"{residual:.2f}")

        if (residual > max_residual): #Get the max value
            max_residual = residual


    print("--------")

    print(f"Max Residual: {max_residual:.2f}") 

    print("--------")

    MAE = np.mean(np.abs(residuals))
    print(f"Mean Absolute Error: {MAE:0.2f}")
    print(f"Accuracy: {100 - MAE:0.2f}%")

    print("--------")

    #Get the r^2 Score
    r2 = r2_score(data['Charge'], predictions)

    print(f"R^2 Score: {r2:0.2f}")

#-------------------------------------------- Main Code ----------------------------------------------------#
if __name__ == "__main__":

    data = pd.read_csv('data/electricity_bill.csv')
    print(data)

    model = linearRegression(data) #Create and train the model

    predictions = predict(model, data) #Predict prices(y) based on the trained model with the x values from the dataset

    additional_info(data, predictions) #Print the additional statistical information

    graph(data, predictions)

'''
* Factors for the Bill calculation:
- Weather
- Appliance Usage (Power Consumption)
'''
