# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:52:10 2018

@author: ashok
"""
#import networkx as nx
from xml.dom import minidom
#from nltk.stem import PorterStemmer
from autocorrect import spell

import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
#from stemming.porter2 import stem


#from textblob import TextBlob
#import jgraph
#import numpy as np

#mydoc = minidom.parse('restaurants-trial.xml')
mydoc = minidom.parse('Restaurants_Train.xml')
text = list(mydoc.getElementsByTagName('text'))
data=[]
tokenword=[]
countwords=[]
distinct_words=[]
clean=[]

not_fired_word=[]
sentiwordsinreview=[]
sentiment=[]
tc=0.7
dt=0.9
for elem in text:
    data.append([elem.firstChild.data])

del(text)
df_data=pd.DataFrame(data)   

stop_words = ["mightn't", 'or', 'doesn', 'y', 'should', 'doing', 'after', 'this', 'those', 'with', 'his', 'herself', 'aren', 'which', 'won', 'itself', 'wouldn', 'it', 'has', 'at', 'is', 'further', 'some', 'myself', "you'd", 've', 'was', 'to', 'until', 'no', "shouldn't", 'couldn', 'whom', 'of', 'between', 'from', 'her', 'me', 'its', 'hasn', 'these', 'through', "wasn't", "don't", 'theirs', 'hadn', 'into', 'been', 'again', 'yours', 'she', 'mightn', 'under', 'while', 'just', 'had', 'about', 'below', 'the', 'll', 'hers', "didn't", 'how', 'm', 'such', "hadn't", 'what', 'be', 'did', 'but', 'each', 't', 'by', 'didn', 'before', 'off', 'where', 'most', 'there', 'ourselves', 'our', 'over', "it's", 'will', 'own', 'he', 'why', 'if', 'weren', 'during', 'have', 'ours', 'more', 'other', 'very', 're', 'above', 'are', 'down', 'shan', 'shouldn', 'an', 'too', 'isn', 'mustn', 'you', 'o', "doesn't", 'not', "haven't", 'they', 'both', 'don', 'does', 'only', 'himself', "aren't", 'your', "hasn't", 'haven', 'all', "that'll", "she's", 'out', "needn't", 'on', 'nor', "shan't", 'needn', 'then', 'having', 'so', "wouldn't", 'who', "you're", "isn't", 'as', 'for', 'them', 'i', "you'll", 'my', 'do', 'here', 's', 'ain', 'themselves', 'ma', 'yourselves', 'we', 'being', "mustn't", "you've", 'that', 'now', 'their', 'few', 'up', 'a', 'in', 'can', 'against', 'same', "couldn't", "won't", 'because', 'than', 'd', 'were', 'once', 'him', 'and', 'any', "weren't", 'yourself', 'am', "should've", 'when', 'wasn''!','@','#','"','$','(','.',')']
  
for line in data:
    for word in line:
        words = nltk.word_tokenize(word.lower())
        tokenword.append(words)

df_data['tokenized_words']=tokenword

autocorrect=[]

for lines in tokenword:
    linesword=[]
    for word in lines:
        linesword.append(spell(word))
    autocorrect.append(linesword)
    

df_data['Autocorrect']=autocorrect

words=df_data['Autocorrect']

clean = words.apply(lambda x: [word for word in x if word not in stop_words])

df_data['clean_text']=clean

for words in clean:
    for word in words:
        countwords.append(word)

del(words,word)
    #clean_words =list([text.strip().lower() for text in countwords if not text in stop_words])

for w in countwords:
    if not w in distinct_words:
        distinct_words.append(w)
del(w)

df_occurrence=pd.DataFrame(distinct_words)

oc_table=[0 for i in range(1) for j in range(len(distinct_words))]        
      
for w1 in range(len(distinct_words)):
    for w2 in range(len(countwords)):
        if distinct_words[w1]==countwords[w2]:
            oc_table[w1]+=1

del(w1,w2)

df2=pd.DataFrame(data=0,index=distinct_words[0:],columns=distinct_words[0:])

df2['occurance']=oc_table

for i in range(len(clean)):            
    for j in range(len(clean[i])-1):
        for k in range(j+1, len(clean[i])):
            a=int(i)
            b=int(j)
            c=int(k)
            w1=clean[a][b]
            w2 =clean[a][c]
            if w1 != w2:
                df2.loc[w1, w2] += 1
                #com[w1][w2] += 1
del(w1,w2,i,j,k,a,b,c)

df_occurrence['Occurrence']=oc_table
df2['occurance']=oc_table

graph_df2=pd.DataFrame(data=0,index=distinct_words[0:],columns=distinct_words[0:])
for i in range((len(distinct_words))-1):
    for j in range(i+1,len(distinct_words)):
        w1=distinct_words[i]
        w2=distinct_words[j]
        if w1 != w2:
            graph_df2.loc[w1,w2]=df2[w2][w1]/df2['occurance'][w2]
            #edg_weight.append(weight)
del(w1,w2,i,j)

root_words_food=['bagels', 'cake', 'potato', 'onion', 'garlic', 'burger', 'dishes', 'dish', 'salads', 'food', 'eating', 'eat', 'perks', 'lime', 'ice', 'beer', 'dosa', 'tomato', 'sausage', 'chicken', 'meats', 'traditional', 'ingredients', 'sardines', 'cheese', 'omelet', 'drinks', 'cuisine', 'pizza', 'eaten', 'tasty', 'salmon', 'rice', 'pickles', 'seafoods', 'sandwich', 'tuna', 'fish', 'soup', 'seafoods', 'seafood', 'sushi', 'pastry', 'goat', 'salad', 'pork', 'delicious', 'noodles', 'meal', 'desert', 'nutrient', 'solid_food', 'breads', 'lamb', 'tapas', 'bread', 'egg', 'eggs']
root_words_price=['deal','high','prices','priced', 'budget', 'afford', 'inexpens', 'cheap', 'expense', 'price', 'term', 'damage', 'bargain', 'cost', 'bill','bills', 'buy', 'cash', 'discount', 'invoice', 'off', 'overprice', 'Purchas', 'toll']
root_words_services=['menu','server', 'waitress', 'order', 'staff', 'deliver', 'friend', 'avail', 'help', 'serv', 'service']
root_words_ambience=['ambience','floor', 'place', 'crowd', 'block', 'ambient', 'atmosphere', 'room', 'area']


fired_word_food=[]
fired_word_price=[]
fired_word_services=[]
fired_word_ambience=[]
food=[]
price=[]
services=[]
ambience=[]
sentiment=[]

def set_aspect1(root_words_food):
    activation_val=pd.DataFrame(data=0,index=distinct_words[0:],columns=distinct_words[0:])
    for w1 in root_words_food:
        if w1 in distinct_words:
            activation_val[w1][w1]=1
            if w1 not in fired_word_food:
                fired_word_food.append(w1)
    
    for i in range(len(clean)):
        for j in range(len(clean[i])-1):
            for k in range(j+1, len(clean[i])):
                a=int(i)
                b=int(j)
                c=int(k)
                w1=clean[a][b]
                w2 =clean[a][c]
                if w1 != w2:
                    activation_val.loc[w1,w2]=min((activation_val[w2][w1])+(activation_val[w1][w1])*(graph_df2.loc[w1,w2])*(dt),1)
                if activation_val.loc[w1,w2]>tc:
                    if w2 not in fired_word_food:
                        fired_word_food.append(w2)
        
    return activation_val

def set_aspect2(root_words_price):
    activation_val=pd.DataFrame(data=0,index=distinct_words[0:],columns=distinct_words[0:])
    for w1 in root_words_price:
        if w1 in distinct_words:
            activation_val[w1][w1]=1
            if w1 not in fired_word_price:
                fired_word_price.append(w1)
    
    for i in range(len(clean)):
        for j in range(len(clean[i])-1):
            for k in range(j+1, len(clean[i])):
                a=int(i)
                b=int(j)
                c=int(k)
                w1=clean[a][b]
                w2 =clean[a][c]
                if w1 != w2:
                    activation_val.loc[w1,w2]=min((activation_val[w2][w1])+(activation_val[w1][w1])*(graph_df2.loc[w1,w2])*(dt),1)
                if activation_val.loc[w1,w2]>tc:
                    if w2 not in fired_word_price:
                        fired_word_price.append(w2)
        
    return activation_val
def set_aspect3(root_words_services):
    activation_val=pd.DataFrame(data=0,index=distinct_words[0:],columns=distinct_words[0:])
    for w1 in root_words_services:
        if w1 in distinct_words:
            activation_val[w1][w1]=1
            if w1 not in fired_word_services:
                fired_word_services.append(w1)
    
    for i in range(len(clean)):
        for j in range(len(clean[i])-1):
            for k in range(j+1, len(clean[i])):
                a=int(i)
                b=int(j)
                c=int(k)
                w1=clean[a][b]
                w2 =clean[a][c]
                if w1 != w2:
                    activation_val.loc[w1,w2]=min((activation_val[w2][w1])+(activation_val[w1][w1])*(graph_df2.loc[w1,w2])*(dt),1)
                if activation_val.loc[w1,w2]>tc:
                    if w2 not in fired_word_services:
                        fired_word_services.append(w2)
        
    return activation_val
def set_aspect4(root_words_ambience):
    activation_val=pd.DataFrame(data=0,index=distinct_words[0:],columns=distinct_words[0:])
    for w1 in root_words_ambience:
        if w1 in distinct_words:
            activation_val[w1][w1]=1
            if w1 not in fired_word_ambience:
                fired_word_ambience.append(w1)
    
    for i in range(len(clean)):
        for j in range(len(clean[i])-1):
            for k in range(j+1, len(clean[i])):
                a=int(i)
                b=int(j)
                c=int(k)
                w1=clean[a][b]
                w2 =clean[a][c]
                if w1 != w2:
                    activation_val.loc[w1,w2]=min((activation_val[w2][w1])+(activation_val[w1][w1])*(graph_df2.loc[w1,w2])*(dt),1)
                if activation_val.loc[w1,w2]>tc:
                    if w2 not in fired_word_ambience:
                        fired_word_ambience.append(w2)
        
    return activation_val


food=set_aspect1(root_words_food)
price=set_aspect2(root_words_price)
services=set_aspect3(root_words_services)
ambience=set_aspect4(root_words_ambience)
df = pd.DataFrame(columns=['Sentances','Category','Sentiment'])

Sentances=[]
category=[]
final_sentiment=""
   

def sentiment_analysis(review):

    import nltk
    from nltk.corpus import sentiwordnet as swn
            
# Part of Speach tagging to each word 
    tagged_texts = nltk.pos_tag(review)
    final_sentiment=""
# creating an empty variable for storing the senti words and their POS score
    ss_set = []
    pos_score=neg_score=token_count=final_score=0
# loop for identifing and storing POS score into impty set
    for word, tag in tagged_texts:
        if 'NN' in tag and list(swn.senti_synsets(word, 'n')):
            ss_set = list(swn.senti_synsets(word, 'n'))[0]
        elif 'VB' in tag and list(swn.senti_synsets(word, 'v')):
            ss_set = list(swn.senti_synsets(word, 'v'))[0]
##            print(ss_set)
        elif 'JJ' in tag and list(swn.senti_synsets(word, 'a')):
            ss_set = list(swn.senti_synsets(word, 'a'))[0]
##            print(ss_set)
        elif 'RB' in tag and list(swn.senti_synsets(word, 'r')):
            ss_set = list(swn.senti_synsets(word, 'r'))[0]
##       print(ss_set)
# calculating the full score of the review
        if ss_set:
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()

# Counting the total words available in the set for sentiment score              
            token_count += 1
            final_score = pos_score - neg_score
##              print(final_score)
##              print(token_count)
            norm_final_score = round(float(final_score) / token_count, 2)
            if norm_final_score>0:
                if neg_score>=1:
                    final_sentiment = 'negative'
                if neg_score>1:
                    final_sentiment = 'conflict'
                else:
                    final_sentiment = 'positive'
            elif norm_final_score<0:
                if pos_score>=1:
                    final_sentiment = 'Positive'
                if pos_score>1:
                    final_sentiment = 'conflict'
                else:
                    final_sentiment = 'negative'
            elif norm_final_score==0:
                if pos_score>neg_score:
                    final_sentiment = 'Positive'
                if neg_score>pos_score:
                    final_sentiment = 'negative'
                else:
                    final_sentiment = 'neutral'
    return final_sentiment

def aspectfortext(plaintext, data):
    sentiment_food=[]
    sentiment_ambience=[]
    sentiment_price=[]
    sentiment_service=[]
  
    for i in range(len(plaintext)):
        word=plaintext[i]
        if ambience.loc[word][word]==1:
            a=1 
            sentiment_ambience.append(word)
            for j in range(i+1,len(plaintext)):
                word2=plaintext[j]
                if word2 in fired_word_ambience:
                    sentiment_ambience.append(word2)
                    a+=float(ambience.loc[word][word2])
            if a>=1:
                Sentances.append(data)
                category.append("ambience")
                sentiment.append(sentiment_analysis(sentiment_ambience))
        elif ambience.loc[word][word]!=1:
            a=ambience.loc[word][word]
            if word in fired_word_ambience:
                sentiment_ambience.append(word)
                    
            for j in range(i+1,len(plaintext)):
                word2=plaintext[j]
                if word2 in fired_word_ambience:
                    a+=float(ambience.loc[word][word2])
                    sentiment_ambience.append(word2)
            if a>=1:
                Sentances.append(data)
                category.append("ambience")
                sentiment.append(sentiment_analysis(sentiment_ambience))

        if food.loc[word][word]==1:
            a=1
            sentiment_food.append(word)
            for j in range(i+1,len(plaintext)):
                word2=plaintext[j]
                if word2 in fired_word_food:
                    a+=float(food.loc[word][word2])
                    sentiment_food.append(word2)
                a+=float(food.loc[word][word2])
            
            if a>=1:
                Sentances.append(data)
                category.append("food")
                sentiment.append(sentiment_analysis(sentiment_food))
        elif food.loc[word][word]!=1:
            a=food.loc[word][word]
            if word in fired_word_food:
                sentiment_food.append(word)
            for j in range(i+1,len(plaintext)):
                word2=plaintext[j]
                if word2 in fired_word_food:
                    a+=float(food.loc[word][word2])
                    sentiment_food.append(word2)
            if a>=1:
                Sentances.append(data)
                category.append("food")
                sentiment.append(sentiment_analysis(sentiment_food))

        if price.loc[word][word]==1:
            a=1 
            sentiment_price.append(word)
            for j in range(i+1,len(plaintext)):
                word2=plaintext[j]
                if word2 in fired_word_price:
                    a+=float(price.loc[word][word2])
                    sentiment_price.append(word2)
            if a>=1:
                Sentances.append(data)
                category.append("price")
                sentiment.append(sentiment_analysis(sentiment_price))
        elif price.loc[word][word]!=1:
            a=price.loc[word][word]
            if word in fired_word_price:
                sentiment_price.append(word)
            for j in range(i+1,len(plaintext)):
                word2=plaintext[j]
                if word2 in fired_word_price:
                    a+=float(price.loc[word][word2])
                    sentiment_price.append(word2)
            if a>=1:
                Sentances.append(data)
                category.append("price")
                sentiment.append(sentiment_analysis(sentiment_price))

        if services.loc[word][word]==1:
            a=1 
            sentiment_service.append(word)
            for j in range(i+1,len(plaintext)):
                word2=plaintext[j]
                if word2 in fired_word_services:
                    sentiment_service.append(word2)
                    a+=float(services.loc[word][word2])
            if a>=1:
                Sentances.append(data)
                category.append("service")
                sentiment.append(sentiment_analysis(sentiment_service))
        elif services.loc[word][word]!=1:
            a=services.loc[word][word]
            if word in fired_word_services:
                sentiment_service.append(word)
            for j in range(i+1,len(plaintext)):
                word2=plaintext[j]
                if word2 in fired_word_services:
                    sentiment_service.append(word2)
                    a+=float(services.loc[word][word2])
            if a>=1:
                Sentances.append(data)
                category.append("service")
                sentiment.append(sentiment_analysis(sentiment_service))
        else:
            Sentances.append(data)
            restset=[]
            for j in range(i+1,len(plaintext)):
                word2=plaintext[j]
                restset.append(word2)
            category.append("anecdotes/miscellaneous")
            sentiment.append(sentiment_analysis(restset))
    if data not in Sentances:
        Sentances.append(data)
        category.append("anecdotes/miscellaneous")
        sentiment.append(sentiment_analysis(plaintext))

for line in range(len(clean)):
    sentance=data[line]
    plaintext=clean[line]
    aspectfortext(plaintext,sentance)
        

df['Sentances']=Sentances
df['Category']=category
df['Sentiment']=sentiment


writer = pd.ExcelWriter('Approachoutput.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()


def result():
    import pandas as pdr

    resultdata = pdr.ExcelFile("Approachoutput.xlsx")
    resultdata.sheet_names
    [u'Sheet1']
    dfnew = resultdata.parse("Sheet1")

    dfnew["is_duplicate"]= dfnew.duplicated()
    
    NewData=pd.DataFrame(dfnew.loc[dfnew['is_duplicate']==False])
    finalreseult_df=pd.DataFrame(NewData)
    new=finalreseult_df.drop('is_duplicate',axis=1)

    finalreseult_df=pdr.DataFrame(new)

    writer = pdr.ExcelWriter('FinalResult17051811.xlsx')
    finalreseult_df.to_excel(writer,'Sheet1')
    writer.save()

result()