import pandas as pd


def apply_func(df,col,func,rel_cols):
    """
        apply function to grid of a column
        Input:
            df - the data frame
            col - column name, a string
            func - the function you want to apply to the grid
            rel_cols - the columns you need as paramter to function, a list of strings
        Return Value:
            ret_series - the return series of given column
    """
    tar_list=df[col]
    return pd.Series(func(tar_list,df[rel_cols]))



df_preprocessed_hair_dryer = pd.read_csv("processed_hair_dryer.csv")
df_preprocessed_microwave = pd.read_csv("processed_microwave.csv")
df_preprocessed_pacifier = pd.read_csv("processed_pacifier.csv")

df_preprocessed_hair_dryer["review_date"]=pd.to_datetime(df_preprocessed_hair_dryer["review_date"],infer_datetime_format=True)
df_preprocessed_microwave["review_date"]=pd.to_datetime(df_preprocessed_microwave["review_date"],infer_datetime_format=True)
df_preprocessed_pacifier["review_date"]=pd.to_datetime(df_preprocessed_pacifier["review_date"],infer_datetime_format=True)


df_preprocessed_hair_dryer=df_preprocessed_hair_dryer.sort_values("review_date")
df_preprocessed_microwave=df_preprocessed_microwave.sort_values("review_date")
df_preprocessed_pacifier=df_preprocessed_pacifier.sort_values("review_date")



df_preprocessed_hair_dryer=df_preprocessed_hair_dryer.reset_index(drop=True)
df_preprocessed_microwave=df_preprocessed_microwave.reset_index(drop=True)
df_preprocessed_pacifier=df_preprocessed_pacifier.reset_index(drop=True)


hair_culmulative_weighted_rating=df_preprocessed_hair_dryer["weighted_ratings"].tolist()
microwave_culmulative_weighted_rating=df_preprocessed_microwave["weighted_ratings"].tolist()
pacifier_culmulative_weighted_rating=df_preprocessed_pacifier["weighted_ratings"].tolist()




ind=0
for item in hair_culmulative_weighted_rating:
    if ind:
        hair_culmulative_weighted_rating[ind]=hair_culmulative_weighted_rating[ind]+hair_culmulative_weighted_rating[ind-1]
    ind=ind+1

ind=0
for item in microwave_culmulative_weighted_rating:
    if ind:
        microwave_culmulative_weighted_rating[ind]=microwave_culmulative_weighted_rating[ind]+microwave_culmulative_weighted_rating[ind-1]
    ind=ind+1


ind=0
for item in pacifier_culmulative_weighted_rating:
    if ind:
        pacifier_culmulative_weighted_rating[ind]=pacifier_culmulative_weighted_rating[ind]+pacifier_culmulative_weighted_rating[ind-1]
    ind=ind+1

# print(pacifier_culmulative_weighted_rating)


df_preprocessed_hair_dryer["culmulative_weighted_rating"]=pd.Series(hair_culmulative_weighted_rating)
df_preprocessed_microwave["culmulative_weighted_rating"]=pd.Series(microwave_culmulative_weighted_rating)
df_preprocessed_pacifier["culmulative_weighted_rating"]=pd.Series(pacifier_culmulative_weighted_rating)



df_preprocessed_hair_dryer.to_csv("processed_hair_dryer.csv")
df_preprocessed_microwave.to_csv("processed_microwave.csv")
df_preprocessed_pacifier.to_csv("processed_pacifier.csv")
