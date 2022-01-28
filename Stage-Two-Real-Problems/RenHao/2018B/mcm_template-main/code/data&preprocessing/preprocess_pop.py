import pandas as pd

	
df1=pd.read_csv("GDP&Population.CSV") # Be sure to pre-preprocessing the data correctly (adjusting format)

df1=df1.sort_values("Country",ascending=True) # sort by country alphabetically


df2=pd.read_csv("GDP&Population2.CSV")

df2=df2.sort_values("Country",ascending=True)

df3=pd.read_csv("BirthRate.csv")


df4=pd.read_csv("Death_Rate.csv")


"""
    select the common country that's in top 50 during two time period
"""
df1cut=df1[df1["Country"].isin(df2["Country"])]
df2cut=df2[df2["Country"].isin(df1["Country"])]

df3cut=df3[df3["Country"].isin(df1cut["Country"])]

df4cut=df4[df4["Country"].isin(df1cut["Country"])]

df=pd.merge(df1cut,df2cut,on="Country")

df=pd.merge(df,df3cut,on="Country")

df=pd.merge(df,df4cut,on="Country")


print(df)
df.to_csv("PreProcessedPoplulation.csv")
