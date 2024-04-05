#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#supress warnings
import warnings
warnings.filterwarnings("ignore")


# In[20]:


iris_df = pd.read_csv("D:\oasis infobyte\Iris.csv")
print("the data has been successfully loaded.")


# In[21]:


iris_df


# In[58]:


iris_df.shape


# In[23]:


iris_df.info()


# In[24]:


iris_df.describe()


# In[26]:


iris_df.isnull().sum()


# In[27]:


print("unique number of values in dataset species:", iris_df["Species"].nunique())
print("Unique Species in iris dataset:", iris_df["Species"].unique())


# In[31]:


sns.pairplot(iris_df, hue = "Species", markers="x")
plt.show()


# In[32]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', data=iris_df, hue='Species')
plt.subplot(1,2,2)
sns.scatterplot(x='SepalWidthCm', y='PetalWidthCm', data=iris_df, hue='Species')
plt.show()


# In[39]:


iris_df["Species"].value_counts().plot(kind="pie", autopct="%1.1f%%", shadow=True, figsize=(5,5))
plt.title("Percentage value in each Species", fontsize=12, c="g")
plt.ylabel("", fontsize=10, c="r")
plt.show()


# In[46]:


plt.figure(figsize=(15,5))
plt.subplot(2,2,1)
sns.barplot(x="Species", y="SepalLengthCm", data=iris_df, palette=("Spectral"))
plt.title("Bar plot SepalLengthCm Vs Species")

plt.subplot(2,2,2)
sns.boxplot(x="Species", y="SepalLengthCm", data=iris_df, palette=("Spectral"))
plt.title("Box plot SepalLengthCm Vs Species")

plt.subplot(2,2,3)
sns.barplot(x="Species", y="SepalLengthCm", data=iris_df, palette=("Spectral"))
plt.title("Bar plot SepalLengthCm Vs Species")

plt.subplot(2,2,4)
sns.boxplot(x="Species", y="SepalLengthCm", data=iris_df, palette=("Spectral"))
plt.title("Box plot SepalLengthCm Vs Species")

plt.show()


# In[48]:


plt.figure(figsize=(15,5))
plt.subplot(2,2,1)
sns.barplot(x="Species", y="SepalLengthCm", data=iris_df, palette=("coolwarm"))
plt.title("Bar plot SepalLengthCm Vs Species")

plt.subplot(2,2,2)
sns.boxplot(x="Species", y="SepalLengthCm", data=iris_df, palette=("coolwarm"))
plt.title("Box plot SepalLengthCm Vs Species")

plt.subplot(2,2,3)
sns.barplot(x="Species", y="SepalLengthCm", data=iris_df, palette=("coolwarm"))
plt.title("bar plot SepalLengthCm Vs Species")

plt.subplot(2,2,4)
sns.boxplot(x="Species", y="SepalLengthCm", data=iris_df, palette=("coolwarm"))
plt.title("Box plot SepalLengthCm Vs Species")

plt.show()


# In[51]:


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
sns.distplot(iris_df["SepalLengthCm"], color="y").set_title("Sepal Length Interval")

plt.subplot(2,2,2)
sns.distplot(iris_df["SepalWidthCm"], color="r").set_title("Sepal Width Interval")

plt.subplot(2,2,3)
sns.distplot(iris_df["PetalLengthCm"], color="g").set_title("Petal Length Interval")

plt.subplot(2,2,4)
sns.distplot(iris_df["PetalWidthCm"], color="b").set_title("Petal Width Interval")

plt.show()


# In[52]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
iris_df["Species"]=le.fit_transform(iris_df["Species"])
iris_df.head()


# In[53]:


iris_df["Species"].unique()


# In[55]:


x=iris_df.iloc[:,[0,1,2,3]]
x.head()


# In[56]:


y=iris_df.iloc[:,[-1]]
y.head()


# In[60]:


print(x.shape)
print(y.shape)


# In[63]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=0)


# In[64]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

