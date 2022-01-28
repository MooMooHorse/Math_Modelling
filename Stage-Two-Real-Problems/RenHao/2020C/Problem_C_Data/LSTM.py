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
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use','br','and','to','a','it','We','It','would','the','one','This','A','2','The'])


    for list_member in list_of_list_of_tokens:
        for stop_item in stop_words:
            while stop_item in list_member:
                list_member.remove(stop_item)

    return list_of_list_of_tokens

def LDA(list_of_list_of_tokens,htmlname):
    from gensim import corpora, models

    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use','br','and','to','a','it','We','It','would','the','one','This','A','2','The'])


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



df_microwave= df_microwave.sort_values("product_id")

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
    df_part=df_microwave[df_microwave["product_id"] == product]
    df_len=[*range(len(df_part))]
    df_part.index=df_len # "reindexing"

    for row_n in df_len:
        row_en=df_part[df_part.index==row_n]
        row_list=[list_of_list_of_tokens[element_ind]]
        row_list.extend(row_en["star_rating"])
        row_list.extend(row_en["helpful_votes"])
        row_list.extend(row_en["total_votes"])
        row_list.extend(row_en["verified_purchase"])
        array_item.append(row_list)
        element_ind=element_ind+1
    
    array_for_LSTM.append(array_item)
    if sample_DEBUG == 1: # debug
        print(array_item)
        df_part.to_csv("check_part.csv")
        sample_DEBUG=0

    
"""
    api:
        array_for_LSTM - a list of array to train
        array_for_LSTM[i] - the array to train
        array_for_LSTM[i][j] - the j-th item of an array
        for every item it looks like
        [[string containing all the words except for stop words],star rating,total votes,verified purchase]
        Example:
            print(array_for_LSTM[0][0])
            output:
                [['I', 'love', 'problem', 'popcorn', 'button', 'set', 
                'long', 'enough', 'complete', 'popping', 'popcorn', 
                'half', 'cooked', 'convection', 'works', 'great', 
                'much', 'simpler', 'previous', 'combination', 'Convection', 
                'Microwave', 'oven'],
                 4, 6, 8, 'N']
"""

print(array_for_LSTM[0][0])


# list_of_list_of_tokens=[]
# for s in df_pacifier["review_body"].astype(str):
#     list_of_tokens = re.findall(r'\w+', s)
#     list_of_list_of_tokens.append(list_of_tokens)
# LDA(list_of_list_of_tokens,'pacifier_html_file.html')



# df_hair.to_csv("hair_dryer_clean.csv")
# df_microwave.to_csv("microwave_clean.csv")
# df_pacifier.to_csv("pacifier_clean.csv")