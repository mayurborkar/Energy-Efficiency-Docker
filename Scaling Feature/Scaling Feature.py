#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv(r"C:\Users\Lenovo\PycharmProjects\EnergyEfficiency\Dataset\Scaling.csv")
df.head()


# In[3]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[4]:


df.head()


# In[5]:


scaling_feature =[feature for feature in df.columns if feature not in ['cooling load','heating load']]
df[scaling_feature].head()


# In[6]:


scaler = MinMaxScaler()
data = scaler.fit_transform(df[scaling_feature])
data


# In[7]:


data.shape


# In[8]:


VIF = pd.DataFrame()
VIF['vif'] = [variance_inflation_factor (data,i) for i in range(data.shape[1])]
VIF['feature'] = df[scaling_feature].columns
VIF


# **As we see that from VIF the value will be beyond 10. So we decide to perform an analysis designed for highly correlated variables, such as Principal Components Analysis**

# In[9]:


pca=PCA()
data2=pca.fit_transform(data)
data2


# In[10]:


data2.shape


# In[11]:


VIF = pd.DataFrame()
VIF['vif'] = [variance_inflation_factor (data2,i) for i in range(data2.shape[1])]
VIF['feature'] = df[scaling_feature].columns
VIF


# In[12]:


one = df[['heating load','cooling load']]


# In[13]:


two = pd.DataFrame(data2,columns=df[scaling_feature].columns)


# In[14]:


df_new = pd.concat([one,two],axis=1)
df_new.head()


# In[15]:


#df_new.to_csv(r'C:\Users\Lenovo\PycharmProjects\EnergyEfficiency\Dataset\Model_Building.csv')


# In[16]:


import pickle
file = 'Energy_Efficiency_pca.pkl'

pickle.dump(pca, open(file,'wb'))


# In[ ]:




