# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 19:14:44 2018

@author: Rohit
"""
from __future__ import print_function
import os
import json
from nltk.corpus import stopwords
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from num2words import num2words
from bs4 import BeautifulSoup

#Inverted Index-> term:List pair where first element of list is Document Freq and subsequent elements are list of DOC-ID,Term freq

file_name_ID={}                                                                     #To store (Document ID: File Name) pair
inverted_index={}                                                                   #To store inverted index
words_list=[]                                                                       #To store all the tokens (for word cloud)    
path='D:/MTECH/SEM 2/Information Retrieval/Assignments/Assignment 2/stories'
stop_words=set(stopwords.words('english'))
porter = PorterStemmer()

#Method to generate Word Cloud from input String
def word_cloud_generator(words_string):
    wordcloud=WordCloud(relative_scaling=1.0,width=800,height=600).generate(words_string)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    
#Method to caclulate the Document frequency of terms in inverted index    
def document_frequency(inverted_index):
    for key,value in inverted_index.iteritems():
        doc_freq=[]
        doc_freq.append(len(value))
        inverted_index[key].insert(0,doc_freq)

#Method to read files from Directory, to include title, to perform preprocessing and to construct inverted index
def read_files(path):
    i=0
    for root,dirs,files in os.walk(path,topdown=False):
                for name in files:
                    directory=root.split('/')[-1]
                    directory=directory.replace('\\','/')
                    file_name_ID[i]=(directory+'/'+name)
                    complete_file_path=os.path.join(root,name)
                    header_processed_string=file_preprocess(complete_file_path,name)
                    processed_string=re.sub("(?<=\\d),(?=\\d)", "",header_processed_string)                #To remove commas within the numbers
                    processed_words=re.sub('([^A-Za-z0-9]+)',' ',processed_string)                         #To remove all special symbols
                    words=word_tokenize(processed_words)                                                  
                    if(name in file_name_title):
                        title=file_name_title[name]
                        title_string=title.split()
                        for l in range(20):                                                                #Increase of title frequency 
                            words.extend(title_string)
                    for word in words:
                        if (word not in stop_words):                                                       # Removing Stop Words
                            word=word.lower()                                                              # Normalization(convert to lower Case)
                            try:
                                stemmed_word=porter.stem(word)                                              #Stemming
                                stemmed_word = unicode(stemmed_word, errors='ignore')
                                try:
                                     val = int(stemmed_word)
                                     stemmed_word=num2words(val)                                           #To convert number to words
                                except ValueError:
                                     pass
                            except:
                                pass
                            words_list.append(stemmed_word)
                            if (stemmed_word not in inverted_index):
                                inverted_index[stemmed_word]=[]
                            if ((len(inverted_index[stemmed_word])==0) or (i != inverted_index[stemmed_word][-1][0])):
                                temp=[]
                                temp.append(i);
                                temp.append(1);
                                inverted_index[stemmed_word].append(temp)
                            else:
                                inverted_index[stemmed_word][-1][1]+=1
                    i+=1
                    

#To remove the Text before the file Blank Line in all the Documents
def file_preprocess(input_path_file,name):
    lines=open(input_path_file,'r').readlines()
    for i,line in enumerate(lines):
        line=line.strip()
        if not line:
            break
    input_file_list= open(input_path_file,'r').readlines()[i+1:]
    complete_file_string=''.join(input_file_list)
    return complete_file_string

#TO extract the Title from index.html file and store write to index.txt file
def parse_index_file():
     for root,dirs,files in os.walk(path,topdown=False):
                for name in files:
                    directory=root.split('/')[-1]
                    directory=directory.replace('\\','/')
                    file_name=(directory+'/'+name)
                    if(name.split('.')[0]=='index'):
                         file_path=os.path.join(root,name)
                         with open(file_path, 'r') as myfile:
                             file_data=myfile.read().replace('\n', '')
                         soup = BeautifulSoup(file_data, 'html.parser')
                         if(file_name=='stories/archive/index.html'):
                             data = soup.find_all('tr')[0].get_text(separator='~!')
                         if(file_name=='stories/index.html'):
                             data = soup.find_all('tr')[3].get_text(separator='~!')
                         elif(file_name=='stories/SRE/index.html'):
                             data = soup.find_all('tr')[0].get_text(separator='~!')
                         #print(data)
                         file_name_title_size=data.split('~!')
                         file_name_title_size=filter(lambda a: str(a) != ' ', file_name_title_size)
                         for i in range(len(file_name_title_size)):
                             if i%3==2:
                                 file_name_title[file_name_title_size[i-2]]=file_name_title_size[i]
                                 with open(name.split('.')[0]+'.txt','a') as file:
                                     file.write(file_name_title_size[i-2]+'  '+file_name_title_size[i]+'\n')
                                     
file_name_title={}                                                                      #To store the (File Name: Title) pair  
parse_index_file()                                                                              
read_files(path)
document_frequency(inverted_index)
word_string=' '.join(words_list)
word_cloud_generator(word_string)
print("Number of Documents: ",len(file_name_ID))    
print("Number of Tokens: ",len(inverted_index))

with open('inverted_index.json','w') as file:                                            #To store the inverted index in json file
    json.dump(inverted_index,file)

with open('doc_id_doc_name.json','w') as file:
    json.dump(file_name_ID,file)

