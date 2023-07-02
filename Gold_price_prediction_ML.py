#!/usr/bin/env python
# coding: utf-8

# # IMPORTING NECESSARY LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# # READING THE FILE

# In[2]:


df=pd.read_csv('E:\DA_PROJECTS\SAMPLE_PRO\gold.csv')
df


# In[3]:


#DISPLAYING FIRST 5 ENTRIES
df.head()


# In[4]:


#DISPLAYING LAST 5 ENTRIES
df.tail()


# In[5]:


#SHOWING TOTAL NUMBER OF ENTRIES
len(df)


# In[6]:


#DISPLAYING THE INFORMATION ABOUT TABLE DATA
df.info()


# In[7]:


#DISPLAIYNG NUMBER OF ROWS AND COLUMNS
df.shape


# In[8]:


#CHECKING ON PRESENCE OF NULL VALUE
df.isnull().sum()


# In[9]:


#STATISTICAL DESCRIPTION OF THE DATA
df.describe().transpose()


# # CREATING CORRELATION BETWEEN PARAMETERS

# In[10]:


correlation=df.corr()


# In[11]:


#PLOTTING THE CORRELATION
plt.figure(figsize= (8,8))
sns.heatmap(correlation, cbar=True,square=True, fmt='.1f',annot=True,annot_kws={'size':8},cmap='Greens')


# In[12]:


#PRINTING THE CORRELATION OF 'GOLD' WITH RESPECT TO OTHER PARAMETERS
print(correlation['GLD'])


# # VISUALIZATION OF DATA USING SEABORN

# In[13]:


#PLOTTING THE PRICE OF GOLD
sns.displot(df['GLD'],color='Blue')


# In[14]:


#PLOTTING GOLD PRICE WITH RESPECT TO IT'S DENSITY
sns.distplot(df['GLD'],color='Blue')


# In[15]:


#EXTRACTING THE REQUIRED COLUMNS
x=df.drop(['Date','GLD'],axis=1)
y=df['GLD']


# # SPLITTING OF TEST AND TRAINING DATASET

# In[16]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=2)


# # RANDOM FOREST REGRESSOR MODEL

# In[17]:


regressor=RandomForestRegressor(n_estimators=100)


# In[18]:


#TRAINING MODEL DATA
regressor.fit(x_train,y_train)


# In[19]:


#PREDICTION ON TEST DATA
test_data_prediction=regressor.predict(x_test)


# In[20]:


print(test_data_prediction)


# In[21]:


#R squared error
error_score=metrics.r2_score(y_test,test_data_prediction)
print("ERROR PERCENTAGE = ",error_score)


# # COMPARING ACTUAL AND PREDICTED VALUES

# In[22]:


#CONVERTING DATA INTO LIST TYPE
y_test=list(y_test)


# In[23]:


#VISUALIZATION OF ACTUAL AND PREDICTED  GOLD PRICE
plt.plot(y_test,color='black',label='Actual value')
plt.plot(test_data_prediction,color='yellow',label='Predicted value')
plt.title('ACTUAL PRICE VS PREDICTED PRICE')
plt.xlabel('Number of values')
plt.ylabel('Gold Price')
plt.legend()
plt.show()

