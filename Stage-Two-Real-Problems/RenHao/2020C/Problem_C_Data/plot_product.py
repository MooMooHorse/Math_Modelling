import pandas as pd
import matplotlib.pyplot as plt

product_hair=pd.read_csv("product_hair.csv")
product_microwave=pd.read_csv("product_microwave.csv")
product_pacifier=pd.read_csv("product_pacifier.csv")

# product_microwave.groupby("product_id").agg({'count':'sum'}).to_csv('tmp.csv')

microwave_cut=product_microwave.groupby("product_id").agg({'count':'sum'})
microwave_cut=microwave_cut[microwave_cut["count"]>75]
microwave_cut=microwave_cut.index.tolist()

product_microwave=product_microwave[product_microwave["product_id"].isin(microwave_cut)]

product_microwave["review_date"]=pd.to_datetime(product_microwave["review_date"])
# print(product_microwave)

product_microwave.plot.scatter(x="product_id",y="review_date",s="count")

plt.show()