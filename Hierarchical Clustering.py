#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd


# In[3]:


#loading data
data1 = pd.read_csv('df_expression.csv')
data1


# In[4]:


data1=data1.drop(['Batch'],axis=1)
data1


# In[5]:


data1=data1.drop(['Unnamed: 0'],axis=1)
data1


# In[6]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[7]:


linkage_data = linkage(data1, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.rc('xtick',labelsize=0.1)
plt.xticks(rotation=90)

fig1 = plt.gcf()
plt.show()
plt.draw()
#fig1.savefig('clustering expression.pdf', dpi=100)


# In[8]:


#dataset after batch correction
data2=pd.read_csv("df_corrected.csv")
data2


# In[10]:


data2=data2.drop(['Unnamed: 0'],axis=1)
data2


# In[11]:


linkage_data = linkage(data2, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.rc('xtick',labelsize=0.1)
plt.xticks(rotation=90)

fig2 = plt.gcf()
plt.show()
plt.draw()
#fig2.savefig('clustering corrected.pdf', dpi=100)

