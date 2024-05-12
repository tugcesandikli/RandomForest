#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[7]:


data=pd.read_csv("/Users/tugcesandikli/Desktop/data (7).csv")


# In[8]:


data.info()


# In[9]:


data.drop(["id","Unnamed: 32"],axis=1,inplace=True)


# In[10]:


data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]


# In[11]:


y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)


# In[12]:


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[13]:


x


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)


# In[15]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Decision Tree Score: " ,dt.score(x_test,y_test))


# In[16]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)


# In[17]:


print("Random Forest Algorithm result: ",rf.score(x_test,y_test))

