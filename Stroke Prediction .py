#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


# linear algebra
import numpy as np
# data processing, CSV file I/O (for example pd.read_csv)
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("stroke prediction data.csv")
df.head()


# # Data Preprocessing

# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.describe()


# # Null Values Treatment

# In[6]:


df.isnull()


# In[28]:


df.isnull().sum()


# In[33]:


sns.distplot(df['stroke'])


# In[9]:


df['bmi'].describe()


# In[10]:


df['bmi'].fillna(df['bmi'].mean(),inplace=True)


# In[11]:


df.isnull().sum()


# In[39]:


get_ipython().system('pip install plotly.express')


# In[40]:


import plotly.express as px

count = df.stroke.value_counts()
#print(count) #checking the value to map the good label
colors=px.colors.sequential.RdBu
fig = px.pie(count, names=['True', 'False'], values=count.values, color_discrete_sequence=colors)
fig.show()


# # Exploratory Data Analysis

# In[12]:


df.drop('id',axis=1,inplace=True)


# In[13]:


df.head()


# In[14]:


cols = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status', 'bmi', 'stroke']]
cols.head()


# In[15]:


##never somoked! strange..


# In[16]:


for column in cols:
    sns.distplot(cols[column])
    plt.show()


# In[ ]:


##formerly smoked is not shown in the result for loop w sns! Why?! i need your help!


# In[ ]:


cols = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']]
cols.head()


# In[ ]:


for column in cols:
    sns.distplot(cols[column])
    plt.show()


# In[ ]:


fig,axes = plt.subplots(4,2,figsize = (16,16))
sns.set_style('darkgrid')
fig.suptitle("Count plot for various categorical features")
sns.countplot(ax=axes[0,0],data=df,x='gender')
sns.countplot(ax=axes[0,1],data=df,x='hypertension')
sns.countplot(ax=axes[1,0],data=df,x='heart_disease')
sns.countplot(ax=axes[1,1],data=df,x='ever_married')
sns.countplot(ax=axes[2,0],data=df,x='work_type')
sns.countplot(ax=axes[2,1],data=df,x='Residence_type')
sns.countplot(ax=axes[3,0],data=df,x='smoking_status')
sns.countplot(ax=axes[3,1],data=df,x='stroke')
plt.show()


# In[15]:


df.groupby('gender').mean()[['age', 'stroke']]


# # Men have less stroke than women!

# In[19]:


sns.violinplot(x='gender', y='age', data=df, hue='stroke')


# In[20]:


sns.violinplot(x='gender', y='age', data=df, hue='stroke', split=True)


# In[25]:


sns.swarmplot(x='gender', y='age', data=df, hue='stroke')


# In[27]:


sns.violinplot(x='gender', y='age', data=df)
sns.swarmplot(x='gender', y='age', data=df, color='black')


# # Work Type

# In[17]:


df['work_type'].unique()


# In[18]:


df['work_type'].value_counts()


# In[19]:


sns.countplot(data=df, x='work_type', hue='stroke')
plt.show()


# In[20]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[21]:


df.plot(kind='box', figsize=(12,9))
plt.show()


# # Label Encoding

# In[22]:


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


# In[23]:


gender=enc.fit_transform(df['gender'])


# In[24]:


smoking_status=enc.fit_transform(df['smoking_status'])


# In[25]:


work_type=enc.fit_transform(df['work_type'])
Residence_type=enc.fit_transform(df['Residence_type'])
ever_married=enc.fit_transform(df['ever_married'])


# In[26]:


df['work_type']=work_type
df['ever_married']=ever_married
df['Residence_type']=Residence_type
df['smoking_status']=smoking_status
df['gender']=gender


# In[27]:


df.head()


# In[28]:


df.info()


# In[30]:


X=df.drop('stroke',axis=1)


# In[31]:


X.head()


# In[32]:


Y=df['stroke']


# In[33]:


Y


# # Splitting data  for train and test

# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)


# In[35]:


X_train


# In[36]:


Y_train


# In[37]:


X_test


# In[38]:


Y_test


# # Normalization

# In[40]:


df.describe()


# In[41]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[42]:


X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)


# In[43]:


X_train_std


# In[45]:


X_test_std


# # Logistic Regression

# In[47]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[48]:


lr.fit(X_train_std,Y_train)


# In[49]:


Y_pred_lr=lr.predict(X_test_std)


# In[50]:


Y_pred_lr[:7]


# # Evaluation for Logistic Regression

# In[51]:


from sklearn.metrics import classification_report, accuracy_score


# In[52]:


print(classification_report(Y_test, Y_pred_lr))


# In[53]:


ac_lr=accuracy_score(Y_test,Y_pred_lr)


# In[54]:


ac_lr


# # Random Forest

# In[55]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[57]:


rf.fit(X_train_std,Y_train)


# In[58]:


Y_pred=rf.predict(X_test_std)


# In[59]:


Y_pred


# In[60]:


Y_pred[:10]


# # Evaluation for Random Forest

# In[61]:


ac_rf=accuracy_score(Y_test,Y_pred)
ac_rf


# In[62]:


plt.bar(['Logistic','Random Forest'],[ac_lr,ac_rf])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.show()


# # Images

# In[63]:


from urllib.request import urlretrieve


# In[65]:


urlretrieve('https://sarhfiles.blob.core.windows.net/live/images/default-source/our-services/stroke-center/befast-800x480378240c3-a1f3-4504-a928-b6c42ef89593.jpg?sfvrsn=325ce5dd_4', 'befast.jpg');


# In[66]:


from PIL import Image


# In[67]:


img = Image.open('befast.jpg')


# In[68]:


img_array = np.array(img)


# In[69]:


img_array.shape


# In[70]:


plt.imshow(img);


# In[71]:


plt.grid(False)
plt.axis('off')
plt.imshow(img_array[120:320,100:300]);

