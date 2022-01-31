import pandas as pd
import numpy as np

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
    stop_words.extend(['1','3','4','5','6','7','8','9'])

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


for product in products_microwave:
    array_item=[]
    df_part=df_microwave[df_microwave["product_id"] == product].copy()

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
        a = (map(lambda x: x.lower(), list_of_list_of_tokens[element_ind]))
        row_list = [list(a)]
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


def Cook_data(array_for_LSTM,del_word_freq=70,Format_Words="Word_List",if_train_rating=True,
if_train_helpful_vote=True,if_train_total_vote=True,if_train_vine=True,if_train_diff_date=True,remove_date=False,
del_product_review_num_threshold=-1
):
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
            You MUST!!!!!!!!!!! insure the date difference is at the end.
    """

    # Filter Products

    for product_index,product in enumerate(array_for_LSTM):
        if len(product)<=del_product_review_num_threshold:
            del array_for_LSTM[product_index]

    # Filter Words using del_word_freq
    dict_for_words={}
    dict_for_word_index={}
    for product_index,product in enumerate(array_for_LSTM):
        for review_index,review in enumerate(product):
            for word_in_review in review[0]:
                if word_in_review in dict_for_words:
                    dict_for_words[word_in_review]+=1
                else:
                    dict_for_words[word_in_review]=0

    dict_for_words=dict(filter(lambda elem: elem[1]>=del_word_freq,dict_for_words.items()))

    for word_ind,dic_key in enumerate(dict_for_words.keys()):
        dict_for_word_index[dic_key]=word_ind


    for product_index,product in enumerate(array_for_LSTM):
        for review_index,review in enumerate(product):
            word_list=[]
            for word_in_review in review[0]:
                if word_in_review in dict_for_words:
                    word_list.append(word_in_review)
            review[0]=word_list

    # print(len(dict_for_word_index),len(dict_for_words))
    Review_mean=[]
    Review_std=[]


    if Format_Words == "Encoded_List":
        for product_index,product in enumerate(array_for_LSTM):
            for review_index,review in enumerate(product):
                word_list=[0]*len(dict_for_words)
                for word_in_review in review[0]:
                    if word_in_review in dict_for_word_index:
                       word_list[dict_for_word_index[word_in_review]]+=1
                review[0]=word_list
    # Normalize every number

        for word_index in range(len(dict_for_words)):
            list_of_n_th_word=[review[0][word_index] for product in array_for_LSTM for review in product]
            list_of_n_th_word=np.array(list_of_n_th_word)
            mean_list=list_of_n_th_word.mean()
            std_dev_list=list_of_n_th_word.std()

            Review_mean.append(mean_list)
            Review_std.append(std_dev_list)

            if std_dev_list!=0:
                list_of_n_th_word=(list_of_n_th_word-mean_list)/std_dev_list
            
            rev_index=0
            for product in array_for_LSTM:
                for review in product:
                    review[0][word_index]=list_of_n_th_word[rev_index]
                    rev_index+=1

        # Package the first dimension
        Review_mean=[Review_mean]
        Review_std=[Review_std] 
    
    if if_train_rating:
        list_of_train_data=[review[1] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)
        mean_list=list_of_train_data.mean()
        std_dev_list=list_of_train_data.std()

        Review_mean.append(mean_list)
        Review_std.append(std_dev_list)

        # print(mean_list,std_dev_list)

        if std_dev_list!=0:
            list_of_train_data=(list_of_train_data-mean_list)/std_dev_list
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                review[1]=list_of_train_data[rev_index]
                rev_index+=1

    if if_train_helpful_vote:
        list_of_train_data=[review[2] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)
        mean_list=list_of_train_data.mean()
        std_dev_list=list_of_train_data.std()

        Review_mean.append(mean_list)
        Review_std.append(std_dev_list)

        if std_dev_list!=0:
            list_of_train_data=(list_of_train_data-mean_list)/std_dev_list
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                review[2]=list_of_train_data[rev_index]
                rev_index+=1

    if if_train_total_vote:
        list_of_train_data=[review[3] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)
        mean_list=list_of_train_data.mean()
        std_dev_list=list_of_train_data.std()

        Review_mean.append(mean_list)
        Review_std.append(std_dev_list)

        if std_dev_list!=0:
            list_of_train_data=(list_of_train_data-mean_list)/std_dev_list
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                review[3]=list_of_train_data[rev_index]
                rev_index+=1

    map_Y_N={}
    map_Y_N['Y']=1
    map_Y_N['N']=0
    for product in array_for_LSTM:
        for review in product:
            review[4]=map_Y_N[review[4]]

    if if_train_vine:
        list_of_train_data=[review[4] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)
        mean_list=list_of_train_data.mean()
        std_dev_list=list_of_train_data.std()

        Review_mean.append(mean_list)
        Review_std.append(std_dev_list)

        if std_dev_list!=0:
            list_of_train_data=(list_of_train_data-mean_list)/std_dev_list
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                review[4]=list_of_train_data[rev_index]
                rev_index+=1
    
    removed_date=[]
    if remove_date:
        for product in array_for_LSTM:
            for review in product:
                removed_date.append(review[5])
                del review[5]
    
    if if_train_diff_date:
        list_of_train_data=[review[-1] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)
        mean_list=list_of_train_data.mean()
        std_dev_list=list_of_train_data.std()

        Review_mean.append(mean_list)
        Review_std.append(std_dev_list)
        
        if std_dev_list!=0:
            list_of_train_data=(list_of_train_data-mean_list)/std_dev_list
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                review[-1]=list_of_train_data[rev_index]
                rev_index+=1

    return array_for_LSTM,removed_date,dict_for_word_index,Review_mean,Review_std

# print(array_for_LSTM[0][0])
array_for_LSTM,removed_date,dict_for_words,Review_mean,Review_std= \
Cook_data(array_for_LSTM,70,"Encoded_List",True,True,True,True,True,True,-1)


def Reverse_Cook(array_for_LSTM,removed_date,dict_for_word_index,Review_mean,Review_std,Format_Words="Word_List",if_train_rating=True,
if_train_helpful_vote=True,if_train_total_vote=True,if_train_vine=True,if_train_diff_date=True,remove_date=False):
    if remove_date:
        rem_id=0
        for product in array_for_LSTM:
            for review in product:
                review.insert(5,removed_date[rem_id])
                rem_id+=1

    mean_std_ind=0

    if Format_Words == "Encoded_List":

        for word_index in range(len(dict_for_word_index)):
            mean_list=Review_mean[0][word_index]
            std_dev_list=Review_std[0][word_index]

            list_of_n_th_word=[review[0][word_index] for product in array_for_LSTM for review in product]
            list_of_n_th_word=np.array(list_of_n_th_word)

            if std_dev_list!=0:
                list_of_n_th_word=list_of_n_th_word*std_dev_list+mean_list
            
            rev_index=0
            for product in array_for_LSTM:
                for review in product:
                    review[0][word_index]=list_of_n_th_word[rev_index]
                    rev_index+=1



        dict_for_word_index = dict((y,x) for x,y in dict_for_word_index.items())
        # print(dict_for_word_index)
        #RecoverWord
        for product_index,product in enumerate(array_for_LSTM):
            for review_index,review in enumerate(product):
                word_list=[]
                for wd_ind,word_freq in enumerate(review[0]):
                    while word_freq>=0:
                        if word_freq>=0.3:
                            word_list.append(dict_for_word_index[wd_ind])
                        word_freq-=1
                review[0]=word_list

        

    if if_train_rating:
        list_of_train_data=[review[1] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)

        mean_list=Review_mean[1]
        std_dev_list=Review_std[1]

        if std_dev_list!=0:
            list_of_train_data=list_of_train_data*std_dev_list+mean_list
        
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                review[1]=list_of_train_data[rev_index]
                rev_index+=1

    if if_train_helpful_vote:
        list_of_train_data=[review[2] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)

        mean_list=Review_mean[2]
        std_dev_list=Review_std[2]

        if std_dev_list!=0:
            list_of_train_data=list_of_train_data*std_dev_list+mean_list
        
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                review[2]=list_of_train_data[rev_index]
                rev_index+=1

    if if_train_total_vote:
        list_of_train_data=[review[3] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)

        mean_list=Review_mean[3]
        std_dev_list=Review_std[3]

        if std_dev_list!=0:
            list_of_train_data=list_of_train_data*std_dev_list+mean_list
        
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                review[3]=list_of_train_data[rev_index]
                rev_index+=1


    if if_train_vine:
        list_of_train_data=[review[4] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)

        mean_list=Review_mean[4]
        std_dev_list=Review_std[4]

        if std_dev_list!=0:
            list_of_train_data=list_of_train_data*std_dev_list+mean_list
        
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                if list_of_train_data[rev_index]>=0.5:
                    review[4]='Y'
                else:
                    review[4]='N'
                rev_index+=1
    
    
    if if_train_diff_date:
        list_of_train_data=[review[-1] for product in array_for_LSTM for review in product]
        list_of_train_data=np.array(list_of_train_data)

        mean_list=Review_mean[-1]
        std_dev_list=Review_std[-1]

        if std_dev_list!=0:
            list_of_train_data=list_of_train_data*std_dev_list+mean_list
        
        rev_index=0
        for product in array_for_LSTM:
            for review in product:
                review[-1]=list_of_train_data[rev_index]
                rev_index+=1

    return array_for_LSTM
# print(Review_mean)
array_for_LSTM=Reverse_Cook(array_for_LSTM,removed_date,dict_for_words,Review_mean,Review_std,"Encoded_List",True,True,True,True,True,True)
# print(array_for_LSTM[0][0])

from tqdm import tqdm
from numpy import array
from sklearn.preprocessing import LabelEncoder

input_size=209
output_size=input_size
train_window=10
train_windows=train_window*input_size
hidden_size=500
words_min_repeatence=70
preseved_space=5
max_value_of_each_preseved_space=[-100]*6
max_value_of_each_preseved_space[3]=1

dic_for_words=[]
real_dic={}
k=-1
for i in range(len(array_for_LSTM)):
    i=k+1
    if i == len(array_for_LSTM):
        break
    if len(array_for_LSTM[i]) > train_window:
        for j in array_for_LSTM[i]:
            dic_for_words.extend(j[0])
            for l in range(len(j[1:])):
                try:
                    if j[l+1]>max_value_of_each_preseved_space[l]:
                        max_value_of_each_preseved_space[l]=j[l+1]
                except:
                    continue
    else:
        del array_for_LSTM[i]
        i-=1
    k=i

print(max_value_of_each_preseved_space)
values = array(dic_for_words)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
real_dic=real_dic.fromkeys(dic_for_words,0)
for i in dic_for_words:
    real_dic[i]+=1
real_dic=dict((k,integer_encoded[dic_for_words.index(k)]) for k,v in real_dic.items() if v>=words_min_repeatence)
j=0
for i in real_dic.keys():
    real_dic[i]=j
    j+=1

reversed_dic={v:k for k,v in real_dic.items()}


dic_for_words=[]
i=0
j=0
k=0
values=[]
label_encoder=[]
integer_encoded=[]

'''
list_for_words=[[],[]]
list_for_words[0]=list(real_dic.keys())
list_for_words[1]=list(real_dic.values())
list_for_words = list(map(list, zip(*list_for_words)))
list_for_words.sort(key=lambda x:x[1],reverse=True)
print((list_for_words))
'''



import torch
import torch.nn as nn
import torch.optim as optim

input_dataset=[]
data_for_single_comment=[]
data_for_each_brand=[]

def create_inout_sequences(input_data, tw,input_size):
    inout_seq = []
    L = len(input_data)
    for i in range(0,L-tw-input_size+1,input_size):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+input_size]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


for i in tqdm(array_for_LSTM):
    data_for_each_brand=[]
    for j in i:
        data_for_single_comment=[0.0]*(input_size-preseved_space)
        for k in j[0]:
            try:
                data_for_single_comment[real_dic[k]]=1.0
            except:
                continue
        for l in range(6):
            if l == 3:
                data_for_single_comment.append(1-float(YorN.index(j[l+1])))
            elif l != 4:
                data_for_single_comment.append(j[l+1]/max_value_of_each_preseved_space[l])
        data_for_each_brand.extend(data_for_single_comment)
    data_for_each_brand=torch.FloatTensor(data_for_each_brand).cuda()
    input_dataset.extend(create_inout_sequences(data_for_each_brand,train_windows,input_size))

print(input_dataset[1][1])
        




class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=200, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size).cuda()

        self.linear = nn.Linear(hidden_layer_size, output_size).cuda()

        self.ReLU = nn.ReLU()

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).cuda(),
                            torch.zeros(1,1,self.hidden_layer_size).cuda())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(train_window ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(train_window, -1))
        predictions = self.ReLU(predictions)
        return predictions[-1]

model = LSTM(input_size,hidden_size,output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for i in tqdm(range(epochs)):
    for seq, labels in input_dataset:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).cuda(),
                            torch.zeros(1, 1, model.hidden_layer_size).cuda())

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%5 == 4:
        print(f'epoch: {i+1:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i+1:3} loss: {single_loss.item():10.10f}')
torch.save(model, f'LSTM_model_{input_size-preseved_space}_{train_window}_{hidden_size}_{epochs}.pth') 
# print(array_for_LSTM[1][1])

# list_of_list_of_tokens=[]
# for s in df_pacifier["review_body"].astype(str):
#     list_of_tokens = re.findall(r'\w+', s)
#     list_of_list_of_tokens.append(list_of_tokens)
# LDA(list_of_list_of_tokens,'pacifier_html_file.html')



# df_hair.to_csv("hair_dryer_clean.csv")
# df_microwave.to_csv("microwave_clean.csv")
# df_pacifier.to_csv("pacifier_clean.csv")