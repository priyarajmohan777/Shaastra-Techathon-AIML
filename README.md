### AIM:





### PROCEDURE:




### CODE:
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

```

```
dataset1=pd.read_csv('/content/train_data_covid.csv')
dataset2=pd.read_csv('/content/test_data_covid.csv')
new_column_data = list(range(1, 3111))
dataset2.to_csv('testset.csv', index=False)
dataset2['Deaths'] = new_column_data
encoder=LabelEncoder()
combined_dataset = pd.concat([dataset1, dataset2], ignore_index=True)
combined_dataset['State/UnionTerritory'] = encoder.fit_transform(combined_dataset['State/UnionTerritory'])
dataset1_encoded = combined_dataset.iloc[:15001]
dataset2_encoded = combined_dataset.iloc[15001:]
dataset1_encoded.to_csv('/content/train_data_covid.csv', index=False)
dataset2_encoded.to_csv('/content/test_data_covid.csv', index=False)
```

```
train=pd.read_csv('/content/train_data_covid.csv')
import datetime as dt
train['Date'] = pd.to_datetime(train['Date'])
train['Date']=train['Date'].map(dt.datetime.toordinal)
train['ConfirmedForeignNational']=train['ConfirmedForeignNational'].replace('-',0)
train['ConfirmedIndianNational']=train['ConfirmedIndianNational'].replace('-',0)
# Transform the same categorical columns in the second dataset
train.head(15000)
```

```
test=pd.read_csv('/content/test_data_covid.csv')
import datetime as dt
test['Date'] = pd.to_datetime(test['Date'])
test['Date']=test['Date'].map(dt.datetime.toordinal)
test['ConfirmedIndianNational']=test['ConfirmedIndianNational'].replace('-',0)
test['ConfirmedForeignNational']=test['ConfirmedForeignNational'].replace('-',0)
test.head()
```

```
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import random
from sklearn.ensemble import GradientBoostingRegressor

X_train=train[['Confirmed','PopulationDensityPerSqKm','Date','Cured','State/UnionTerritory']]
y_train=train['Deaths']

x_test=test[['Confirmed','PopulationDensityPerSqKm','Date','Cured','State/UnionTerritory']]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaling to the testing data
X_test_scaled = scaler.transform(x_test)

poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X_train_scaled)
x_test_poly = poly_features.transform(X_test_scaled)
model=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=38, random_state=22)
model.fit(X_poly,y_train)
# Make predictions on the test data
y_pred = model.predict(x_test_poly)
act = y_train.head(3110)
# Calculate mean squared error
mse = mean_squared_error(act, y_pred)
print("Mean Squared Error:",mse)
```

```
test['Deaths']=y_pred
test['Deaths'] = test['Deaths'].apply(lambda x: max(0, x))
test['Deaths'] = test['Deaths'].apply(lambda x: round(x))
# Write the updated DataFrame back to a CSV fi
test
```

```
predictions = pd.DataFrame(test, columns=['Sno','Deaths'])
predictions.to_csv('sample_submission.csv', index = False)
```

### RESULT:



