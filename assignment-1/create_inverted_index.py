# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:23:31 2018

@author: Rohit
"""
from __future__ import print_function
"""This File reads the documents, performs preprocessing on the document and words and creates an inverted index 
for the terms in the document.
##Correct path of the document must be specified before running 
 """
import os
from nltk.corpus import stopwords
import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#For plotting the word cloud
def word_cloud_generator(words_string):
    wordcloud=WordCloud(relative_scaling=1.0,width=800,height=600).generate(words_string)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
#Everything before the first blank line is removed while processing the document
def header_preprocess(input_path_file):
    lines=open(input_path_file,'r').readlines()
    for i,line in enumerate(lines):
        line=line.strip()
        if not line:
            break
    input_file_list= open(input_path_file,'r').readlines()[i+1:]
    complete_file_string=''.join(input_file_list)
    return complete_file_string

tokenizer= RegexpTokenizer(r'\w{3,}')                                           #To all the words of length 3 or more
porter = PorterStemmer()                                                        #Stemming on the words using porter's algorithm                                              
stop_words=set(stopwords.words('english'))

inverted_index={}
file_name_ID={}

path='D:/MTECH/SEM 2/Information Retrieval/Assignments/Assignment 1/20_newsgroups'  #Path to the document collection
i=0
words_list=[]
for root,dirs,files in os.walk(path,topdown=False): 
            for name in files:
                directory=root.split("\\",1)[1]
                file_name_ID[i]=directory+'/'+name
                path_file=os.path.join(root,name)
                header_processed_string=header_preprocess(path_file)
                words=tokenizer.tokenize(header_processed_string)               # To remove punctuation mark, comma etc and to form tokens
                for word in words:
                    if (word not in stop_words):                                # Removing Stop Words
                        word=word.lower()                                       # Normalization
                        try:
                            stemmed_word=porter.stem(word)
                            stemmed_word = unicode(stemmed_word, errors='ignore')
                        except:
                            pass
                        words_list.append(stemmed_word)
                        if (stemmed_word not in inverted_index):
                            inverted_index[stemmed_word]=[]
                        if i not in inverted_index[stemmed_word]:
                            inverted_index[stemmed_word].append(i)
                i+=1

words_string=' '.join(words_list)
word_cloud_generator(words_string)            
print("Number of Documents: ",len(file_name_ID))    
print("Number of Tokens: ",len(inverted_index))

with open('inverted_index.json','w') as file:
    json.dump(inverted_index,file)

with open('doc_id_doc_name.json','w') as file:
    json.dump(file_name_ID,file)

