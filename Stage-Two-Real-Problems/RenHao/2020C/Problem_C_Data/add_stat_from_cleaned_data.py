import pandas as pd

df_hair = pd.read_csv("hair_dryer_clean.csv")
df_microwave=pd.read_csv("microwave_clean.csv")
df_pacifier=pd.read_csv("pacifier_clean.csv")
"""
    sample checks if +1 works
    sample=df_hair[0:10]

    print(sample)

    sample["weighted_votes"]=2*sample["helpful_votes"]-sample["total_votes"]+1
"""

df_hair["weighted_votes"]=2*df_hair["helpful_votes"]-df_hair["total_votes"]+1
df_microwave["weighted_votes"]=2*df_microwave["helpful_votes"]-df_microwave["total_votes"]+1
df_pacifier["weighted_votes"]=2*df_pacifier["helpful_votes"]-df_pacifier["total_votes"]+1

df_hair["weighted_ratings"]=(df_hair["star_rating"]-3)*df_hair["weighted_votes"]
df_microwave["weighted_ratings"]=(df_microwave["star_rating"]-3)*df_microwave["weighted_votes"]
df_pacifier["weighted_ratings"]=(df_pacifier["star_rating"]-3)*df_pacifier["weighted_votes"]

"""
    sample campare
    print(sample)
"""
df_hair.to_csv("hair_dryer_clean.csv")
df_microwave.to_csv("microwave_clean.csv")
df_pacifier.to_csv("pacifier_clean.csv")



# print(df_hair)