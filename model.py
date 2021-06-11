import numpy as np
import pandas as pd
data=pd.read_csv("E:\Deployment\data.csv")
data.drop(data.columns[[0]], axis=1, inplace=True)
data.info()
# check column wise missing values
data.isnull().sum()
#replace missing values with maximum value of combination of VB and Height.
data["Coverage"]=data.groupby(["Height","VB"])["Coverage"].transform(lambda x:x.fillna(x.max()))
#check again
data.isnull().sum()
# get mean ,count,standard deviation, min,max and 25,50,75 percentile of numeric variables
data.describe()
#check the number of occurance of 10000 in dataset
len(data[data["Coverage"]==10000])
# remove the outliers (row with coverage =10000)
data=data[data["Coverage"]!=10000]
# independent variables which are needed to build the model.
x=data[["Height","VB","Tilt"]]
# put y as dependent variable( coverage)
y=data["Coverage"]
from sklearn.model_selection import train_test_split
# tuple unpacking to seperate train and test data
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=101)
from sklearn.ensemble import RandomForestRegressor
# create regressor object 
model = RandomForestRegressor(n_estimators = 100, random_state = 0)
model.fit(x_train, y_train)
# get predicted value of coverage from the model by using test data. y_pred is predicted value of coverage in test data set.
y_pred = model.predict(x_test)
## to test with new value
new_number= [[20,6,5]]
  
# Create the pandas DataFrame 
data_new = pd.DataFrame(new_number, columns = ['Height', 'VB','Tilt']) 
new_predict=model.predict(new_number)
  
import joblib
joblib.dump(model, 'model.pkl')