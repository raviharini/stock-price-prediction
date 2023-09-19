#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[110]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[111]:


data=pd.read_csv('NSE_GLOBAL_DS')
data.head()


# In[112]:


data.info()


# In[113]:


data.shape


# In[114]:


data.describe()


# In[115]:


data.corr()


# # DATA PREPROCESSING

# In[116]:


data.isnull().sum()


# In[117]:


# replacing null values with the mean of the open column
data['Open'] = data['Open'].fillna(data['Open'].mean())

# replacing null values with the mean of the High column
data['High'] = data['High'].fillna(data['High'].mean())

# replacing null values with the mean of the Low column
data['Low'] = data['Low'].fillna(data['Low'].mean())

# replacing null values with the mean of the Close column
data['Close'] = data['Close'].fillna(data['Close'].mean())

# replacing null values with the mean of the AdjClose column
data['Adj Close'] = data['Adj Close'].fillna(data['Adj Close'].mean())

# replacing null values with the mean of the Volume column
data['Volume'] = data['Volume'].fillna(data['Volume'].mean())


# In[118]:


data.isnull().sum()
# there is no missing values present in this dataset


# In[119]:


# checking whtether it contains duplicates
data.duplicated().sum()


# # DATA VISUALIZATION

# In[120]:


data['Open'].plot(figsize=(16,6))


# In[121]:


features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
 
plt.subplots(figsize=(20,10))
 
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sns.distplot(data[col])
plt.show()


# In[122]:


#Boxplot
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sns.boxplot(data[col])
plt.show()


# In[123]:


#HeatMap
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr() > 0.9, annot=True, cbar=False)
plt.show()


# In[124]:


# this gives the correlation between the variables
corr_matrix = data.corr()
corr_matrix


# In[125]:


# Scatter Plot Between Open and Close
plt.scatter(data['Open'],data['Close'])
plt.xlabel('Open')
plt.ylabel('Close')
plt.title('open and Close')
# open and close are highly correlated


# In[126]:


# Scatter Plot Between Volume and Low
plt.scatter(data['Volume'],data['Close'])
plt.xlabel('Volume')
plt.ylabel('Close')
plt.title('Volume and Close')
# Volume and close are not much correlated


# In[127]:


# #changing date to nomal form
# from datetime import datetime
# solve_date=pd.to_datetime(data['Date'])
# dates=solve_date.dt.year
# data['Date']=dates
# data


# # SPLITTING DATA

# In[128]:


x = data.iloc[:,[1,2,3,6]]
y = data.iloc[:,[4]]


# In[129]:


from sklearn.preprocessing import StandardScaler
name = x.columns
scale=StandardScaler()
x_old=scale.fit_transform(x)
x = pd.DataFrame(x_old , columns=name)
x


# In[130]:


y


# In[131]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[132]:


x_train.shape


# In[133]:


x_train.shape


# In[134]:


x_test.shape


# In[135]:


y_test.shape


# In[136]:


x_test


# # LINEAR REGRESSION

# In[137]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()


# In[138]:


model=LR.fit(x_train,y_train)
model


# In[139]:


pred=model.predict(x_test)
pred


# In[140]:


y_test


# In[141]:


#Accuracy
from sklearn.metrics import r2_score
acc=r2_score(y_test,pred)
acc


# In[142]:


from sklearn import metrics
print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, pred), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, pred), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, pred)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, pred), 4))
errors = abs(pred - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[143]:


prediction_df = x_test.copy()
prediction_df['Close'] = y_test
prediction_df['Predicted Price'] = pred
prediction_df.head()


# In[144]:


plt.scatter(y_test,pred)
plt.xlabel("Actual Values")
plt.ylabel("Prediction")
plt.show()


# # RANDOM FOREST

# In[145]:


from sklearn.ensemble import RandomForestRegressor
models = RandomForestRegressor()
RF = models.fit(x_train,y_train)
RF


# In[146]:


pred = RF.predict(x_test)
pred_y = pd.DataFrame(pred)
pred_y


# In[147]:


#Accuracy
from sklearn.metrics import r2_score
acc=r2_score(y_test,pred_y)
acc


# In[148]:


pred = pred.reshape(899,1)
pred


# In[149]:


prediction_df = x_test.copy()
prediction_df['Close'] = y_test
prediction_df['Predicted Price'] = pred
prediction_df.head()


# In[150]:


print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, pred_y), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, pred_y), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, pred_y)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, pred_y), 4))
errors = abs(pred - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[151]:


plt.scatter(y_test,pred_y)
plt.xlabel("Actual Values")
plt.ylabel("Prediction")
plt.show()


# In[ ]:





# # DECISION TREE

# In[152]:


from sklearn.tree import DecisionTreeRegressor
algo = DecisionTreeRegressor()
DT = algo.fit(x_train, y_train)
DT


# In[153]:


ypred = DT.predict(x_test)
ypreds = pd.DataFrame(ypred)
ypreds


# In[154]:


ypred = ypred.reshape(899,1)
ypred


# In[155]:


#Accuracy
from sklearn.metrics import r2_score
acc=r2_score(y_test,ypreds)
acc


# In[156]:


print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, ypreds), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, ypreds), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, ypreds)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, ypreds), 4))
errors = abs(ypred - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[157]:


plt.scatter(y_test,ypred)
plt.xlabel("Actual Values")
plt.ylabel("Prediction")
plt.show()


# In[ ]:





# In[158]:


import pickle
pickle.dump(models,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.053689, 0.063834, 0.064263, -0.662774]]))

