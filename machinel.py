#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # loading Boston housing data set

# In[2]:


from sklearn.datasets import load_boston


# In[3]:


boston=load_boston()


# In[4]:


boston.keys()


# In[5]:


print(boston.data)


# In[6]:


print(boston.feature_names)


# ## Preparing The Dataset 

# In[7]:


dataset=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[8]:


dataset


# In[9]:


dataset.head()


# In[10]:


dataset.describe()


# In[11]:


dataset['price']=boston.target


# In[12]:


dataset.head()


# In[13]:


dataset.info()


# In[14]:


dataset.isna().sum()


# ## Exploratory data analysis

# In[15]:


dataset.corr()


# In[16]:


import seaborn as sns
sns.pairplot(dataset)


# In[17]:


import seaborn as sns
sns.regplot(x="RM",y="price",data=dataset)


# In[18]:


sns.regplot(x="RM",y="LSTAT",data=dataset)


# In[19]:


sns.regplot(x="price",y="LSTAT",data=dataset)


# In[20]:


x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[21]:


x.head()


# In[22]:


y


# ## Model Building

# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[24]:


X_train


# In[25]:


X_test


# ## Standardize the dataset

# In[26]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[27]:


X_train=scaler.fit_transform(X_train)


# In[28]:


X_test=scaler.transform(X_test)


# In[29]:


X_train


# ## Model Training

# In[31]:


from sklearn.linear_model import LinearRegression


# In[32]:


regression=LinearRegression()


# In[36]:


regression.fit(X_train,Y_train)


# ## Print Coefficient and intercept

# In[40]:


print (regression.coef_)


# In[42]:


print(regression.intercept_)


# In[43]:


reg_pred=regression.predict(X_test)


# In[44]:


reg_pred


# In[ ]:




