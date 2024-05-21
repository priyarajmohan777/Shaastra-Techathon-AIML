## Descripton:
The goal of this challenge is to estimate the death toll from a pandemic that spreads widely, with an emphasis on the Covid-19 outbreak in India. Accurate death count forecasting is intended to support emergency preparedness and resource allocation.

## Key Features:

### Dataset:
 From January 2020 to August 2021, records of Covid-19 instances in India's various states are included in this dataset. It contains data on the quantity of cases, patients who have recovered, and fatalities. Predictive model evaluation and training depend heavily on the dataset.

### Methodology:
  Model selection, hyperparameter tuning, feature engineering, and thorough preprocessing.

### Evaluation statistic: 
  Any pertinent statistic that quantifies the discrepancy between expected and actual deaths, such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE), will be used to evaluate the models' performance.

### Models Explored:
  The following models were investigated: gradient boosting, random forest, decision tree, and linear regression.

### Final Model:
  The final model is a gradient boosting regression with optimized hyperparameters and polynomial features.

### Architecture: 
  The Gradient Boosting Regressor was selected because of its resilience against overfitting and capacity to manage intricate, non-linear data interactions. It creates a strong predictive model by the iterative combination of weak learners, making accurate predictions necessary for healthcare resource allocation.

#### Data Processing:
  ![image](https://github.com/priyarajmohan777/Shaastra-Techathon-AIML/assets/119475942/490fccc8-2e77-4531-a57e-00a0790abcdd)

#### Model Training:
![image](https://github.com/priyarajmohan777/Shaastra-Techathon-AIML/assets/119475942/0dbf5856-47fa-4402-a186-2dcd00177c06)


## The Code for the Model:
https://github.com/priyarajmohan777/Shaastra-Techathon-AIML/blob/main/iit%20final%20report%20.ipynb

## Result:
  The final model proved to be useful in predicting pandemic-related mortality, exhibiting good predictive potential with low errors. Important factors that helped with planning and resource allocation included date, population density, confirmed cases, and cured patients. These factors also played critical roles in the projections.
 

