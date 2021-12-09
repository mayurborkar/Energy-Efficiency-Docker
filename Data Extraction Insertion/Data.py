#!/usr/bin/env python
# coding: utf-8

# In[14]:


from application_logging.logger import App_Logger
from DatabaseConnection.Database import dataBaseOperation
import pandas as pd


# In[2]:


logger = App_Logger('logFiles/Expo & Impo.log')


# ## Using Database energyefficiency

# In[4]:


logger.info('INFO','Using The energyefficiency Database')
connection = dataBaseOperation()
logger.info('INFO','Creating The Connection With The DataStax Connection')


# ## Using Keyspace energy

# In[5]:


logger.info('INFO','Using The energy Keyspace')
connection.useKeySpace()
logger.info('INFO','Using The energy Keyspace For Table Creation')


# ## Creating The Table

# In[6]:


logger.info('INFO','Creating The Table With Name energy_data')
connection.createTable()
logger.info('INFO','Table Is Created Inside The Keyspace having a Name energy_data')


# ## Exporting The ENB2012_data.csv  File Into Database

# In[8]:


logger.info('INFO','Trying To Put The Data Into The Database For Backup Purpose')
connection.insertIntoTable()
logger.info('INFO','The File Upload Into The Database')


# ## Getting The Data From The Database

# In[13]:


logger.info('INFO','Trying To Get The Data Form The Database For EDA Purpose')
print(connection.getData())
logger.info('INFO','The Data Is Import Successfuly')


# In[20]:


logger.info('INFO','Loading The Data Into Given IDE')
df = pd.DataFrame(connection.getData())
df_analysis=df.sort_values('id')
df_analysis


# In[21]:


df_analysis.to_csv('Dataset/Analysis_Purpose.csv')


# In[ ]:




