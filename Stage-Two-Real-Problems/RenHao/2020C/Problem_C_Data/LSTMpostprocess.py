import numpy as np
import pandas as pd

outputs=np.load('outputs.npy',allow_pickle=True)
array_for_train_data=np.load('array_for_10_trainwindows.npy',allow_pickle=True)
outputs=outputs.tolist()
array_for_train_data=array_for_train_data.tolist()
# print(outputs[0][0])
"""
    api:
        array_for_train_data - same structure with array_for_LSTM, except brand with comments number that less than or equals to trainwindows deleted.
            Example:
                print(array_for_train_data[0][0])
                output:
                    [['really', 'like', 'microwave', 'like', 'easy', 'get', 'amazon', 'really', 'good', 'convection', 'oven', 'although', 'somewhat', 'noisy', 'fan', 
                    'easy', 'install', 'came', 'clear', 'instructions', 'amazon', 'delivered', 'quickly', 'best', 'price', 'anywhere'], 
                    5, 6, 10, 'Y', Timestamp('2007-01-04 00:00:00'), 0]
            Note:
                none

        outputs - almost same structure with array_for_LSTM
            outputs[i] - all n predicted comments for ith brand (n = train_windows, i here is the brand order in array_for_train_data rather than the brand order in array_for_LSTM)
            for every item it looks like 
            [[string containing all the words except for stop words],star rating,total votes,verified purchase(1-Yes;0-No),diff_date(in days)]
            Example:
                print(outputs[0][0])
                output:
                    [['mount', 'models', 'paid', 'looking', 'mechanism', 'keypad', 'newly', 'frozen', 'cu', 'close', 'come', 'crazy', 'blue', 'cabinet', 'chat', '58', 
                    '00', '125', '29', '2015', '150', '3rd', '11', '04', '29', '300', '26', '220v', '11', '1200', '2nd', '27', '2014', '14', '17', 'additionally', '23'],
                    3.5709089040756226, 22.98689065501094, 37.08765381574631, 0.7639150619506836, 44.649807915091515]
            Note:
                If (dtype = integer) is preferred, just let me know

"""

def add_empty_date(list_of_products,pos=-1,insert_time="22/2/2022"):# init value is something that's impossible to exist in data
    for product in list_of_products:
        for review in product:
            review.insert(pos,pd.to_datetime(insert_time))
    return list_of_products

outputs=add_empty_date(outputs) # add date

sample_DEBUG=1 #number of item to display for debugging
df=pd.DataFrame([],columns=["product_id_processed","comments","star_rating","helpful_votes","total_votes","verified_purchase","review_date","diff_date"])
product_index=0
for product in array_for_train_data:
    df_product=pd.DataFrame(product,columns=["comments","star_rating","helpful_votes","total_votes","verified_purchase","review_date","diff_date"])
    df_predict=pd.DataFrame(outputs[product_index],columns=["comments","star_rating","helpful_votes","total_votes","verified_purchase","review_date","diff_date"])
    df_product["is_predict"]=0
    df_predict["is_predict"]=1
    df_product["product_id_processed"]=product_index
    df_predict["product_id_processed"]=product_index
    df=pd.concat([df,df_product,df_predict])


    product_index=product_index+1


    if sample_DEBUG!=0 :
        print(df_predict)
        sample_DEBUG=sample_DEBUG-1
df=df.astype({"is_predict":"int64"})
df.index.name="index_product_wise"
# print(df)

df.to_csv("postprocess_out/original.csv")
