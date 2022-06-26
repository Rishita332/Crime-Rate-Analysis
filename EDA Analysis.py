# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 08:32:28 2021

@author: window 10
"""
import numpy as np
import pandas as pd

pd.set_option("display.max_columns" ,12)
pd.set_option("display.max_rows" ,100)

df = pd.read_csv("crime rate analysis.csv")
df
df.dtypes
df.columns

# changing the column name
df.rename(columns = {"Primary Type" : "Crime_type" , 'Description':'Crime_description' , 'Location Description':'Crime_location' }, inplace = True)
df.columns
df.shape
df.isnull().sum()

# chnaging the value of column Arrest and domestic
df['Arrest'].replace({True :"Yes",  False : "No" },inplace = True)
df['Domestic'].replace({True :"Yes",  False : "No" },inplace = True)

df['Crime_description'].head(100)
df['Crime_location'].head(100)
df['Crime_location'].isnull().sum()

# Removing rows of missing values
df = df[df['Crime_location'].notna()]
df.isnull().sum()
df.columns
df.shape

df.dtypes
df.ID.head(10)
df["ID"] = df["ID"].astype("str")
df.dtypes

# Droping the column
df = df.drop(["Updated On"], axis =1 )
df.columns

import datetime
df["Date"]=pd.to_datetime(df["Date"])

df["Month"]=df["Date"].dt.month
df["Day"]=df["Date"].dt.day
df["Hour"]=df["Date"].dt.hour


df.columns
df.head()

df.to_excel("Cleaned_crimeData.xlsx")

# EDA Analysis

# Most crimes likely to occour
df.columns
df
df['Crime_type'].head(20)
df['Crime_type'].value_counts().head()

# Highetst Arrest
h = df[df['Arrest'] == "Yes"]
h
h.groupby('Crime_type').size().count()
h.groupby('Crime_type')["Arrest"].value_counts().sort_values(ascending = False)
h[['Crime_type', 'Arrest']].groupby(["Arrest"]).size()

# Domestic Assult 
Domes = df[df['Domestic'] == "Yes"]
len(Domes)


# Highest crimes in which month and year
df.groupby('Year')['Crime_type'].value_counts().sort_values(ascending = False)
df.groupby('Month')['Crime_type'].value_counts().sort_values(ascending = False)

pd.set_option("display.max_columns" ,12)
pd.set_option("display.max_rows" ,1000)
df.groupby(['Year','Month'])['Crime_type'].size().max()




