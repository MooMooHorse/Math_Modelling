import pandas as pd

df_hair = pd.read_csv("hair_dryer.tsv",sep='\t')
df_microwave=pd.read_csv("microwave.tsv",sep='\t')
df_pacifier=pd.read_csv("pacifier.tsv",sep='\t')

# print(df_hair["verified_purchase"])

YorN=['Y','N']

"""
    check verified purchase : Y/N
"""
df_hair=df_hair[df_hair["verified_purchase"].isin(YorN)]
df_microwave=df_microwave[df_microwave["verified_purchase"].isin(YorN)]
df_pacifier=df_pacifier[df_pacifier["verified_purchase"].isin(YorN)]
"""
    check vine : Y/N
"""
df_hair=df_hair[df_hair["vine"].isin(YorN)]
df_microwave=df_microwave[df_microwave["vine"].isin(YorN)]
df_pacifier=df_pacifier[df_pacifier["vine"].isin(YorN)]

# print(df_microwave)
"""
    check total_votes : if it is numeric
"""
df_hair=df_hair[df_hair["total_votes"].map(str).str.isnumeric()]
df_microwave=df_microwave[df_microwave["total_votes"].map(str).str.isnumeric()]
df_pacifier=df_pacifier[df_pacifier["total_votes"].map(str).str.isnumeric()]

"""
    check helpful_votes : if it is numeric
"""
df_hair=df_hair[df_hair["helpful_votes"].map(str).str.isnumeric()]
df_microwave=df_microwave[df_microwave["helpful_votes"].map(str).str.isnumeric()]
df_pacifier=df_pacifier[df_pacifier["helpful_votes"].map(str).str.isnumeric()]


"""
    check review : if it contains empty strings
"""

def check_empty(s):
    if not s:
        print("empty string!")
        return False
    return True


df_hair=df_hair[df_hair["review_body"].astype(str).apply(check_empty)]
df_microwave=df_microwave[df_microwave["review_body"].astype(str).apply(check_empty)]
df_pacifier=df_pacifier[df_pacifier["review_body"].astype(str).apply(check_empty)]

def Del_list(list_of_list_of_tokens,htmlname):
    from gensim import corpora, models

    import nltk
    # nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use','br','and','to','a','it','We','It','would','the','one','This','A','2','The','I','When'])


    for list_member in list_of_list_of_tokens:
        for stop_item in stop_words:
            while stop_item in list_member:
                list_member.remove(stop_item)

    return list_of_list_of_tokens

def LDA(list_of_list_of_tokens,htmlname):
    from gensim import corpora, models

    import nltk
    # nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use','br','and','to','a','it','We','It','would','the','one','This','A','2','The','I'])


    for list_member in list_of_list_of_tokens:
        for stop_item in stop_words:
            while stop_item in list_member:
                list_member.remove(stop_item)


    # list_of_list_of_tokens = [["a","b","c"], ["d","e","f"]]
    # ["a","b","c"] are the tokens of document 1, ["d","e","f"] are the tokens of document 2...
    dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]

    num_topics = 15
    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                    id2word=dictionary_LDA, \
                                    passes=4, alpha=[0.01]*num_topics, \
                                    eta=[0.01]*len(dictionary_LDA.keys()))
    

    import pyLDAvis
    import pyLDAvis.gensim_models
    vis = pyLDAvis.gensim_models.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
    pyLDAvis.save_html(vis, htmlname)

    return 


import re

# list_of_list_of_tokens=[]
# for s in df_hair["review_body"].astype(str):
#     list_of_tokens = re.findall(r'\w+', s)
#     list_of_list_of_tokens.append(list_of_tokens)

# LDA(list_of_list_of_tokens,'hair_drier_html_file.html')

import numpy as np
df_microwave["review_date"]=pd.to_datetime(df_microwave["review_date"])

df_microwave= df_microwave.sort_values(by=["product_id","review_date"],ascending=[True,True])

df_microwave["diff_date"]=pd.to_datetime(df_microwave["review_date"]).diff().apply(lambda x: x/np.timedelta64(1, 'D')).fillna(0).astype('int64')

df_microwave.groupby("product_id").agg({'star_rating':'count'}).to_csv("number_over_each_product.csv")

products_microwave=df_microwave.groupby("product_id").agg({'star_rating':'count'}).index.tolist()

"""
    tokenize comments
"""
list_of_list_of_tokens=[]
for s in df_microwave["review_body"].astype(str):
    list_of_tokens = re.findall(r'\w+', s)
    list_of_list_of_tokens.append(list_of_tokens)

list_of_list_of_tokens = Del_list(list_of_list_of_tokens,'microwave_html_file.html')

"""
    create array
"""

array_for_LSTM=[]
element_ind=0 # since it's sorted, you can extract element from token_list in this way

sample_DEBUG=0 # turn on <- =1

pd.set_option('mode.chained_assignment', None) # It must be a dickhead to set a warning on this point!

for product in products_microwave:
    array_item=[]
    df_part=df_microwave[df_microwave["product_id"] == product]
    df_part.loc[df_part.index[0],"diff_date"]=0

    df_len=[*range(len(df_part))]
    df_part.index=df_len # "reindexing"


    if sample_DEBUG != 0: # debug
        print(df_part["diff_date"])
        # print(diff_days)
        # print((pd.to_datetime(row_en["review_date"])-last_date).dt.days)
        # print(array_item)
        # df_part.to_csv("check_part.csv")
        sample_DEBUG=sample_DEBUG-1
    for row_n in df_len:
        row_en=df_part[df_part.index==row_n]
        row_list=[list_of_list_of_tokens[element_ind]]
        row_list.extend(row_en["star_rating"])
        row_list.extend(row_en["helpful_votes"])
        row_list.extend(row_en["total_votes"])
        row_list.extend(row_en["verified_purchase"])
        row_list.extend(row_en["review_date"])
        row_list.extend(row_en["diff_date"])


        last_date=pd.to_datetime(row_en["review_date"])


        array_item.append(row_list)
        element_ind=element_ind+1
    

    array_for_LSTM.append(array_item)

    
"""
    api:
        array_for_LSTM - a list of array to train
        array_for_LSTM[i] - the array to train
        array_for_LSTM[i][j] - the j-th item of an array
        for every item it looks like
        [[string containing all the words except for stop words],star rating,total votes,verified purchase,review_date,diff_date(in days)]
        Example:
            print(array_for_LSTM[0][0])
            output:
               [['When', 'remodeling', 'kitchen', 'decided', 'replace', 
               'Sharp', 'countertop', 'convection', 'microwave', 'range', 
               'model', 'I', 'really', 'miss', 'old', 'new', 'model', 'seems',
                'lot', 'noisier', 'old', 'I', 'went', '1', '5', 'cubic', 'feet',
                 '1', '1', 'feels', 'like', 'I', 'lost', 'lot', 'room', 'hood', 
                 'light', 'VERY', 'dim', 'replacement', 'bulbs', 'cost', 'least', 
                 '4', 'piece', 'find', 'Because', 'vent', 'kitchen', 'sent', 'away',
                  'Sharp', 'charcoal', 'filter', 'cost', '20', 'filter', 'plus', 'shipping', 
                  'handling', 'planned', 'vent', 'outside', 'vent', 'even', 'come', 'close', 'matching', 
                  'hole', 'former', 'range', 'hood', 'cooking', 'turntable', 'white', 'keep', 'clean', 
                  'month', 'old', 'stained', 'already', 'I', 'like', 'cooking', 'area', 'eye', 'level', 
                  'I', 'researched', 'price', 'comparable', 'items', 'web', 'buying', 'Amazon', 'offered', 
                  'best', 'price', 'well', 'free', 'shipping', 'I', 'pleased', 'Amazon'],
                3, 56, 61, 'Y', '12/10/2004', 0]
        Note:
            the date difference is in (days)
"""
from tqdm import tqdm
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dic_for_words=[]
real_dic={}
for i in array_for_LSTM:
    for j in i:
        dic_for_words.extend(j[0])
values = array(dic_for_words)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
real_dic=real_dic.fromkeys(dic_for_words,0)
for i in tqdm(dic_for_words):
    real_dic[i]+=1

list_for_words=[[],[]]
list_for_words[0]=list(real_dic.keys())
list_for_words[1]=list(real_dic.values())
list_for_words = list(map(list, zip(*list_for_words)))
list_for_words.sort(key=lambda x:x[1],reverse=True)
print(list_for_words[0:200])




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class lstm(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,output_size=1,num_layer=2):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer)
        self.layer2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x

model = lstm(2,4,1,2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)





'''

# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']


inverted = label_encoder.inverse_transform([argmax(onehot_encoded[:, :])])
print(inverted)

'''
# print(array_for_LSTM[1][1])

# list_of_list_of_tokens=[]
# for s in df_pacifier["review_body"].astype(str):
#     list_of_tokens = re.findall(r'\w+', s)
#     list_of_list_of_tokens.append(list_of_tokens)
# LDA(list_of_list_of_tokens,'pacifier_html_file.html')



# df_hair.to_csv("hair_dryer_clean.csv")
# df_microwave.to_csv("microwave_clean.csv")
# df_pacifier.to_csv("pacifier_clean.csv")