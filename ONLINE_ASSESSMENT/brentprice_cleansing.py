#WQD7005 - DATA MINING
#INSTRUCTOR : PROF. DR. TEH YING WAH!
#ZAIMIE AZMIN BIN ZAINUL ABIDIN (OLD MATRIX : WQD190018 /NEW MATRIX : 17202336)
#ONLINE ASSESSMENT : DATA CLEANSING TASK
#DATE : 08 MAY 2020
#--------------------------------------

import pandas as pd
#Read the crawling data that stored in csv file
df = pd.read_csv('mywebscrapBrentFinal.csv')
#Get the shape of the data row and columns
print(df.shape)

#Get the info of the data row and columns
print(df.info)

#Get the head and tail of the data
print(df.head())
print(df.tail())

#Look for data that is null within the columns
print(df.isnull().sum())

#Check how many rows are not affected by the null value
print(df.dropna(how='any').shape)

#Check mean for all columns
print(df.mean())

#Fill in the null value for each column with mean
df2=df.fillna(df.mean())

#Check the new dataframe that fill with mean
print(df2.isnull().sum())
print (df2)

#Save dataframe into new csv file
df2.to_csv('new_mywebscrapBrentFinal.csv')