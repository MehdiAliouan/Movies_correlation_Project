#!/usr/bin/env python
# coding: utf-8

# In[57]:


# Import Libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[58]:


# import the data and read it 

df = pd.read_csv(r"C:\Users\ElMehdi\Downloads\movies.csv\movies.csv")


# In[59]:


# Inspecting the dataset

df.head()


# In[60]:


df.describe()


# In[61]:


# let's look for any missing data

df.isnull().sum()


# In[62]:


# data type 

df.dtypes


# In[63]:


# Creating correct year column

df['year_released'] = df['released'].astype(str).str[8:13]


# In[64]:


df.head()


# In[65]:


df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[66]:


df.duplicated().sum()


# In[67]:


df.drop_duplicates()


# In[68]:


# let's look for correlation between the data 

# Correlation between Budgets and Gross

plt.scatter(df['budget'], df['gross'])
plt.xlabel('budget')
plt.ylabel('gross revenue')
plt.title('budget vs gross revenue')
plt.show()


# In[69]:


# Correlation between Budgets and Gross with seaborn

sns.regplot(x='budget', y='gross', data=df, scatter_kws ={"color":"red"}, line_kws = {"color": "purple"})


# In[70]:


correlation = df.corr()
print(correlation)


# In[71]:


# Correlation matrix for movie features

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.title('Correlation Matrix for Numeric Features')
plt.show()


# In[72]:


df_numerized = df

for col_name in df_numerized.columns :
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[73]:


plt.figure(figsize=(14,10))
correlation_matrix = df_numerized.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.title('Correlation matrix for features')
plt.show()


# In[75]:


df_numerized.corr()


# In[ ]:


# Votes and Budget have the highest correlation to gross revenues

# Company has low correlation

