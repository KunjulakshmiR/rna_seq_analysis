#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install combat


# In[2]:


import csv
import pandas as pd
import numpy as np
import pickle
from combat.pycombat import pycombat
import matplotlib.pyplot as plt
import joblib


# In[3]:


dataset_1=pd.read_csv("pan_cancer_NORMALBLOOD.csv")
dataset_1=dataset_1.set_index("Gene")
dataset_1.drop_duplicates(inplace=True)
dataset_1=dataset_1.transpose()
dataset_1.insert(0,'Batch',1)
dataset_1


# In[4]:


dataset_2=pd.read_csv("pan_cancer_data_mrna_seq_v2_rsem - data_mrna_seq_v2_rsem.csv")
dataset_2=dataset_2.drop(['Entrez_Gene_Id'],axis=1)
dataset_2=dataset_2.dropna()
dataset_2.rename(columns={"Hugo_Symbol":"Gene"}, inplace=True)
dataset_2=dataset_2.set_index('Gene')
dataset_2.drop_duplicates(inplace=True)
dataset_2=dataset_2.transpose()
dataset_2.insert(0,'Batch',2)
dataset_2


# In[ ]:





# In[5]:


df_expression = pd.concat([dataset_1,dataset_2],axis=0,join="inner")
df_expression


# In[50]:


df_expression.to_csv('df_expression.csv')


# In[6]:


df_expression.dtypes


# In[7]:


cols= df_expression.columns
df_expression[cols] = df_expression[cols].apply(pd.to_numeric, errors='coerce', axis=1)
df_expression.dtypes


# In[8]:


#df_expression['label'] = np.full(df_expression.shape[0],2)


# In[9]:


plt.boxplot(df_expression)
plt.show()
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib','inline')


# In[10]:


batch=[]
datasets=[dataset_1,dataset_2]
for j in range(len(datasets)):
    batch.extend([j for _ in range(len(datasets[j].columns))])


# In[11]:


df_corrected=pycombat(df_expression.drop(columns=["Batch"]).transpose(),df_expression["Batch"]).transpose()
df_corrected


# In[52]:


df_corrected.to_csv('df_corrected.csv')


# In[12]:


plt.boxplot(df_corrected)
plt.show()


# In[13]:


df1=df_corrected.iloc[:32,:]
df1.to_csv('normal.csv')
df1


# In[14]:


df1.columns.values[0] = "RAB8A"
  

df1


# In[15]:


col=df1.loc[:,:]
df1['mean normal']=col.mean(axis=0)


# In[16]:


pd.DataFrame(df1.mean(axis=0))


# In[17]:


df1['normal_mean'] = df1.mean(axis=0)
df1


# In[18]:


df1.describe()


# In[19]:


df2=df_corrected.iloc[32:,:]
df2.to_csv('Pancreatic_cancer.csv')

df2


# In[20]:


file1=df1.describe()
file1


# In[21]:


file2=df2.describe()
file2


# In[22]:


new_N=file1.filter(['Gene','mean'],axis=0)
new_N


# In[23]:


df_N=np.log2(new_N)
df_N


# In[24]:


new_P=file2.filter(['Gene','mean'],axis=0)
new_P


# In[25]:


df_P=np.log2(new_P)
df_P


# In[26]:


df3=df_N.append(df_P,ignore_index=True)
df3


# In[27]:


df3.index=['Mean Normal','Mean Cancer']
df3


# In[28]:


df4=df3.transpose()
df4


# In[29]:




df4.plot(kind='bar',figsize=(5,3))

plt.show
#plt.savefig("pancreatic_cancer_barplot_batch corrected.pdf",format="pdf",bbox_inches="tight")


# In[30]:


#plt.savefig("pancreatic_cancer_barplot_R.pdf",format="pdf",bbox_inches="tight")


# In[31]:


plt.boxplot(df3)
plt.show()


# In[32]:


df3=df_corrected.iloc[32:64,:]
#df3.to_csv('Pancreatic_cancer.csv')

df3


# In[33]:


df4=df3.transpose()


# In[34]:


df4


# In[35]:


#df4.to_csv('Pancreatic_cancer32.csv')


# In[36]:


#df3.to_csv('Pancreatic_Cancer_32_with index')


# In[37]:


#pip install keras


# In[38]:


#pip install tensorflow


# In[39]:


import glob
import numpy as np
#from keras.preprocessing.image import load_img,img_to_array
import os
import matplotlib.pyplot as plt
#import cv2
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(df_expression.iloc[:,0:-1], df_expression.iloc[:,-1], test_size=0.20, random_state=0)


# In[41]:


from sklearn.decomposition import PCA
pca_2D = PCA(n_components=2).fit_transform(x_train)


# In[ ]:





# In[42]:


mycolors=["r","b"]
labelTups = ['normal','Pan Cancer']
label=y_train
for i,mycolor in enumerate(mycolors):
        plt.scatter(pca_2D[label == i, 0],
                    pca_2D[label == i, 1], color=mycolor)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(labelTups, loc='upper right')
plt.title('PCA(n_component=2)')
plt.show()


# In[43]:


x_tr, x_te, y_tr, y_te = train_test_split(df_corrected, df_expression.iloc[:,-1], test_size=0.20, random_state=0)


# In[44]:


D=df_expression.iloc[:,-1]
D


# In[45]:


#x_trm, x_te, y_tr, y_te = train_test_split(df_corrected, df_expression, test_size=0.20, random_state=0)


# In[46]:


from sklearn.decomposition import PCA
PCA_2D = PCA(n_components=2).fit_transform(x_tr)


# In[47]:


PCA_2D


# In[48]:


mycolors=["r","b"]
labelTups = ['Normal','Pan Cancer']
label=y_tr
for i,mycolor in enumerate(mycolors):
        plt.scatter(PCA_2D[label == i, 0]
                    , PCA_2D[label == i, 1],color=mycolor)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(labelTups, loc='upper right')
plt.title('PCA(n_component=2)')
plt.show()


# In[49]:


A=PCA_2D[label == i, 1]
A


# In[ ]:




