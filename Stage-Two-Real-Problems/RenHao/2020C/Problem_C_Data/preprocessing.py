import pandas as pd

df_hair = pd.read_csv("hair_dryer_clean.csv")
df_microwave=pd.read_csv("microwave_clean.csv")
df_pacifier=pd.read_csv("pacifier_clean.csv")

"""
    short demo to see how group by can be used to enhance performance and optimize implementation
    sample_df=df_microwave[0:10]
    print(sample_df)

    sample_df=sample_df.groupby("review_date")["helpful_votes"].sum()
    print(sample_df)

"""

hair_helpful_culmulative=df_hair.groupby("review_date")["helpful_votes"].sum()
microwave_helpful_culmulative=df_microwave.groupby("review_date")["helpful_votes"].sum()
pacifier_helpful_culmulative=df_pacifier.groupby("review_date")["helpful_votes"].sum()

hair_weighted_helpful_culmulative=df_hair.groupby("review_date")["weighted_votes"].sum()
microwave_weighted_helpful_culmulative=df_microwave.groupby("review_date")["weighted_votes"].sum()
pacifier_weighted_helpful_culmulative=df_pacifier.groupby("review_date")["weighted_votes"].sum()


hair_weighted_rating_culmulative=df_hair.groupby("review_date")["weighted_ratings"].sum()
microwave_weighted_rating_culmulative=df_microwave.groupby("review_date")["weighted_ratings"].sum()
pacifier_weighted_rating_culmulative=df_pacifier.groupby("review_date")["weighted_ratings"].sum()


hair_total_culmulative=df_hair.groupby("review_date")["total_votes"].sum()
microwave_total_culmulative=df_microwave.groupby("review_date")["total_votes"].sum()
pacifier_total_culmulative=df_pacifier.groupby("review_date")["total_votes"].sum()


# print(hair_helpful_culmulative.to_frame())
df_preprocessed_hair_dryer=hair_helpful_culmulative.to_frame().merge(hair_total_culmulative.to_frame(),on="review_date")\
    .merge(hair_weighted_helpful_culmulative.to_frame(),on="review_date")\
    .merge(hair_weighted_rating_culmulative.to_frame(),on="review_date")

df_preprocessed_microwave=microwave_helpful_culmulative.to_frame().merge(microwave_total_culmulative.to_frame(),on="review_date")\
    .merge(microwave_weighted_helpful_culmulative.to_frame(),on="review_date")\
    .merge(microwave_weighted_rating_culmulative.to_frame(),on="review_date")
df_preprocessed_pacifier=pacifier_helpful_culmulative.to_frame().merge(pacifier_total_culmulative.to_frame(),on="review_date")\
    .merge(pacifier_weighted_helpful_culmulative.to_frame(),on="review_date")\
    .merge(pacifier_weighted_rating_culmulative.to_frame(),on="review_date")


df_preprocessed_hair_dryer.to_csv("processed_hair_dryer.csv")
df_preprocessed_microwave.to_csv("processed_microwave.csv")
df_preprocessed_pacifier.to_csv("processed_pacifier.csv")
