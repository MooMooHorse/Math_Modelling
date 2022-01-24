import pandas as pd

df=pd.read_csv("PreProcessedPoplulation.csv")

dt_2020=df.Pop_2020

dt_2050=df.Pop_2050

diff_dt=dt_2050-dt_2020

df.reindex(columns=list(df.columns) + ["Diff_2050_2020"])

df["Diff_2050_2020"]=diff_dt

df.to_csv("PreProcessedSta.csv")