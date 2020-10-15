#!/usr/bin/env python
# coding: utf-8

# # Prediction using supervised machine learning Task done by SHARAD PARMAR
# In this task based upon the number of hours studied we will predict the percentage of marks that the student is expected to score.

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


df=pd.read_csv("task1.csv")


# In[16]:


df


# In[6]:


df.shape


# In[7]:


df.describe()


# In[11]:


df.plot(x="Hours",y="Scores",style='x')
plt.xlabel("Studied Hours")
plt.ylabel("Scores Obtained")
plt.show()


# In[12]:


#feature and target variables
X = df.Hours
y = df.Scores


# In[13]:


#data splitting into test and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[14]:


X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)


# In[21]:


#Training Model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)


# In[27]:


reg.score(X_test, y_test)


# In[35]:


plt.scatter(data.Hours, data.Scores, marker = 'x', color = 'blue')
plt.plot(df.Hours, reg.predict(df[['Hours']]), color = 'red')  #plotting the line of best fit
plt.xlabel('No. of Hours')
plt.ylabel('Scores')


# In[30]:


#evaluation of model

from sklearn import metrics
predictions = reg.predict(X_test)
print('Mean absolute error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean squared error:', metrics.mean_squared_error(y_test, predictions))
print('Root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# Making predictions

# In[33]:


#making predictions
hrs_inp = float(input("Enter hours studied: "))
y_pred = reg.predict([[hrs_inp]])
s = str(y_pred)
print("Predicted Score: {}" .format(s[2:-2]))


# In[ ]:




