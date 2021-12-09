#!/usr/bin/env python
# coding: utf-8

# ### Loading Package & Data

# In[1]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import pickle


# In[2]:


df = pd.read_csv(r"C:\Users\Lenovo\PycharmProjects\EnergyEfficiency\Dataset\Model_Building.csv")
df.head()


# In[3]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[4]:


df.head()


# In[5]:


X = df.drop(['heating load','cooling load'],axis=1)
y1 = df['heating load']
y2 = df['cooling load']


# In[6]:


X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X,y1,y2,test_size=0.33,random_state=20)


# # Model Building 

# In[7]:


Acc = pd.DataFrame(index=None, columns=['model','train_Heating','test_Heating','train_Cooling','test_Cooling'])


# In[8]:


regressors = [['LinearRegression',LinearRegression()],
            ['Lasso',Lasso()],
            ['ElasticNet',ElasticNet()],
            ['RandomForestRegressor',RandomForestRegressor()]]


# In[9]:


for mod in regressors:
    name = mod[0]
    model = mod[1]

    model.fit(X_train,y1_train)
    actr1 = r2_score(y1_train, model.predict(X_train))
    acte1 = r2_score(y1_test, model.predict(X_test))

    model.fit(X_train,y2_train)
    actr2 = r2_score(y2_train, model.predict(X_train))
    acte2 = r2_score(y2_test, model.predict(X_test))
    
    Acc = Acc.append(pd.Series({'model':name, 'train_Heating':actr1,'test_Heating':acte1,'train_Cooling':actr2,'test_Cooling':acte2}),ignore_index=True )
Acc


# # Model Tuning

# In[10]:


param_grid = {'alpha': [0.1, 0.09, 0.04, 0.05, 0.07, 0.004]}

lasso = Lasso(max_iter=50000, tol=0.0000003)
grid_search_lasso = GridSearchCV(lasso, param_grid, 'neg_mean_absolute_error', cv=10)
grid_search_lasso.fit(X_train,y2_train)

print("R-Squared::{}".format(grid_search_lasso.best_score_))
print("Best Hyperparameters::\n{}".format(grid_search_lasso.best_params_))


# In[11]:


lasso = Lasso(alpha=0.004)

lasso.fit(X_train,y1_train)
print("R-Squared on Heating test dataset={}".format(lasso.score(X_test,y1_test)))

lasso.fit(X_train,y2_train)   
print("R-Squaredon Cooling test dataset={}".format(lasso.score(X_test,y2_test)))


# In[12]:


param_grid = {'alpha': [.1,.2,.3,.9,1,5,9,10]}

ridge = Ridge()
grid_search_ridge = GridSearchCV(ridge, param_grid, 'neg_mean_absolute_error', cv=10)
grid_search_ridge.fit(X_train,y2_train)

print("R-Squared::{}".format(grid_search_ridge.best_score_))
print("Best Hyperparameters::\n{}".format(grid_search_ridge.best_params_))


# In[13]:


ridge = Ridge(alpha=0.1)

ridge.fit(X_train,y1_train)
print("R-Squared on Heating test dataset={}".format(ridge.score(X_test,y1_test)))

ridge.fit(X_train,y2_train)   
print("R-Squaredon Cooling test dataset={}".format(ridge.score(X_test,y2_test)))


# In[14]:


param_grid = {'alpha':[0.1,0.2,0.3,0.08,0.06,50,100],'l1_ratio':[.1,.3,.5,.9,1]}

elastic = ElasticNet(tol=0.001)
grid_search_elastic = GridSearchCV(elastic, param_grid, 'neg_mean_absolute_error', cv=10)
grid_search_elastic.fit(X_train,y2_train)

print("R-Squared::{}".format(grid_search_elastic.best_score_))
print("Best Hyperparameters::\n{}".format(grid_search_elastic.best_params_))


# In[15]:


elastic = ElasticNet(alpha=0.06,l1_ratio=1)

elastic.fit(X_train,y1_train)
print("R-Squared on Heating test dataset={}".format(elastic.score(X_test,y1_test)))

elastic.fit(X_train,y2_train)   
print("R-Squaredon Cooling test dataset={}".format(elastic.score(X_test,y2_test)))


# In[16]:


param_grid=[{'n_estimators':[350,400,450], 'max_features':[1,2], 'max_depth':[85,90,95]}]

RFR = RandomForestRegressor(n_jobs=-1)
grid_search_RFR = GridSearchCV(RFR, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search_RFR.fit(X_train, y2_train)

print("R-Squared::{}".format(grid_search_RFR.best_score_))
print("Best Hyperparameters::\n{}".format(grid_search_RFR.best_params_))


# In[17]:


RFR = RandomForestRegressor(n_estimators = 350, max_features = 2, max_depth= 85, bootstrap= True)

RFR.fit(X_train,y1_train)
print("R-Squared on Heating test dataset={}".format(RFR.score(X_test,y1_test)))

RFR.fit(X_train,y2_train)   
print("R-Squaredon Cooling test dataset={}".format(RFR.score(X_test,y2_test)))


# In[18]:


Acc1 = pd.DataFrame(index=None, columns=['model','train_Heating','test_Heating','train_Cooling','test_Cooling'])


# In[19]:


regressors1 = [['Lasso', Lasso(alpha=0.004)],
               ['Ridge', Ridge(alpha=0.1)],
               ['ElasticNet', ElasticNet(alpha=0.06,l1_ratio=1)],
               ['RandomForestRegressor', RandomForestRegressor(n_estimators = 350, max_features = 2, max_depth= 85, 
                                                               bootstrap= True)]]
              


# In[20]:


for mod in regressors1:
    name = mod[0]
    model = mod[1]
    
    model.fit(X_train,y1_train)
    actr1 = r2_score(y1_train, model.predict(X_train))
    acte1 = r2_score(y1_test, model.predict(X_test))
    
    model.fit(X_train,y2_train)
    actr2 = r2_score(y2_train, model.predict(X_train))
    acte2 = r2_score(y2_test, model.predict(X_test))
    
    Acc1 = Acc1.append(pd.Series({'model':name, 'train_Heating':actr1,'test_Heating':acte1,'train_Cooling':actr2,'test_Cooling':acte2}),ignore_index=True )
Acc1


# In[22]:


model_heating = RandomForestRegressor(n_estimators = 350, max_features = 2, max_depth= 85, bootstrap= True)

model_heating.fit(X_train,y1_train)
y1_pred = model_heating.predict(X_test)
actr1 = r2_score(y1_train, model_heating.predict(X_train))
acte1 = r2_score(y1_test, model_heating.predict(X_test))


# In[23]:


print("RandomForestRegressor: R-Squared on train dataset={}".format(actr1))
print("RandomForestRegressor: R-Squared on test dataset ={}".format(acte1))


# In[24]:


model_cooling = RandomForestRegressor(n_estimators = 350, max_features = 2, max_depth= 85, bootstrap= True)

model_cooling.fit(X_train,y2_train)
y2_pred = model_cooling.predict(X_test)
actr2 = r2_score(y2_train, model_cooling.predict(X_train))
acte2 = r2_score(y2_test, model_cooling.predict(X_test))


# In[25]:


print("RandomForestRegressor: R-Squared on train dataset={}".format(actr2))
print("RandomForestRegressor: R-Squared on test dataset ={}".format(acte2))


# In[26]:


file = "Energy_Efficiency_Rf_Heating.pkl"

pickle.dump(model_heating,open(file,'wb'))


# In[27]:


file = "Energy_Efficiency_Rf_Cooling.pkl"

pickle.dump(model_cooling,open(file,'wb'))


# In[28]:


x_ax = range(len(y1_test))
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.plot(x_ax, y1_test, label="Actual Heating")
plt.plot(x_ax, y1_pred, label="Predicted Heating")
plt.title("Heating test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Heating load (kW)')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(x_ax, y2_test, label="Actual Cooling")
plt.plot(x_ax, y2_pred, label="Predicted Cooling")
plt.title("Coolong test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Cooling load (kW)')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)

plt.show()


# In[29]:


def AAD(y1_test, y1_pred):
    AAD =[]
    for i in range(len(y1_pred)):
        AAD.append((y1_pred[i] - y1_test.values[i])/y1_test.values[i]*100)
    return AAD

x_ax = range(len(y1_test))
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.plot(x_ax, AAD(y1_test, y1_pred), label="Relative deviation obtained on Heating load")
plt.title("Heating load")
plt.xlabel('X-axis')
plt.ylabel('Error (%)')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(x_ax, AAD(y2_test, y2_pred), label="Relative deviation obtained on Cooling load")
plt.title("Cooling load")
plt.xlabel('X-axis')
plt.ylabel('Error (%)')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




