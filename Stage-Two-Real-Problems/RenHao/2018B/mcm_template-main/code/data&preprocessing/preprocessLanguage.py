import pandas as pd

df=pd.read_csv("PreProcessedPoplulationbyLanguage.csv")

df=df.sort_values(by="Language",ascending=True)

df=df.groupby("Language").sum()

print(df)

df=df.to_csv("PreProcessedPoplulationbyLanguageGrouping.csv")