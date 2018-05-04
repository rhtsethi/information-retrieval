# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 14:06:40 2018

@author: Rohit
"""

from __future__ import print_function
import json
import math
import sys
import re
import operator
from nltk.metrics import edit_distance 
from num2words import num2words
from nltk.stem import PorterStemmer

with open('inverted_index.json') as f:
    inverted_index=json.load(f)
    
with open('doc_id_doc_name.json') as f:
    file_name_ID=json.load(f)
 
porter=PorterStemmer()
  
#To find the Word from term vocabulary with min edit distance and maximum document frequency      
def spell_correction(query):                                  
    for word in query:
        min_edit=99999
        df_correct_word=0
        for term,posting_list in inverted_index.iteritems():
            ed=edit_distance(term,word)
            if (ed<min_edit) or (ed==min_edit and posting_list[0][0]>df_correct_word):
                min_edit=ed
                new_term=term
                df_correct_word=posting_list[0][0]
        new_corrected_query.append(new_term)

#To calculate the tf-idf score of a query term        
def calc_tf_idf_score_query(word):
    w_t_q=1+math.log10(new_corrected_query.count(word))
    idf_t_q=math.log10((N*1.0)/(inverted_index[word][0][0]+1))                  #1 is added to include query document    
    tf_idf_word=w_t_q*idf_t_q
    tf_idf_query[word]=tf_idf_word

#To calculate the cosine similarity between each query word and document             
def cosine_similarity_score(new_corrected_query):
    for q_term in new_corrected_query:
        posting_list=inverted_index[q_term]
        for i in range(1,len(posting_list)):
            if posting_list[i][0] not in Score:
                Score[posting_list[i][0]]=0
                Length[posting_list[i][0]]=0                                # To initialize the Length dictionary key values to 0
            tf_idf_q_term_doc=((1+math.log10(posting_list[i][1]))*(math.log10((N*1.0)/posting_list[0][0])))
            Score[posting_list[i][0]]+=tf_idf_query[word]*tf_idf_q_term_doc
 
#To calculate the Document Normalisation factor(Length)    
def calc_doc_length():
    for term,posting_list in inverted_index.iteritems():
        for i in range(1,len(posting_list)):
            if posting_list[i][0] in Score:
                Length[posting_list[i][0]]+= (math.pow((1+math.log10(posting_list[i][1]))*(math.log10((N*1.0)/posting_list[0][0])),2))
                
#Caculating the Normalised Score(Dividing by the Legnth)        
def normalised_score():
    for doc_id,score in Score.iteritems():
        Norm_score[doc_id]=(score/math.sqrt(Length[doc_id]))
 
#To check and get the query result from Cache (if present)        
def query_caching(new_corrected_query):
    with open('query_cache_cosine.json') as f:
        query_cache=json.load(f)
    for each_query in query_cache:
        if cmp(new_corrected_query,each_query[0])==0:
            for doc in each_query[1]:
                print (str(doc))
            sys.exit()
            
N=len(file_name_ID)                                      #Number of Documents
query=[]                                                 #query word list
new_corrected_query=[]                                   #Query after processing(Spell correction,stemming,etc)
tf_idf_query={}                                          #Tf-idf value of query terms   
Score={}                                                 #To store the Score(unnormalised)
Length={}                                                #To store the Length of relevant docs
Norm_score={}                                            #To store the Normalised Score
top_k_docs=[]                                            #Top K Documents
query_cache=[]                                           #To store the retrieved result from query cache file
k=5                                                      # Top k documents to be retrieved
arg_count=len(sys.argv)
if arg_count!=1:
    for i in range(1,arg_count):
        query.append(sys.argv[i])
else:
    print("No argument provided!!!")
    sys.exit()

query_string=' '.join(query)
query_string=re.sub("(?<=\\d),(?=\\d)", "",query_string)                #To remove commas within the numbers
query=query_string.split()

for i in range(len(query)):                                             #Porter stemming on query
    query[i]=porter.stem(query[i])

for i in range(len(query)):                                             #To convert number to words
   try:
       val = int(query[i])
       query[i]=num2words(val)
   except ValueError:
       continue

spell_correction(query)
query_caching(new_corrected_query)

with open('query_cache_cosine.json') as f:                              #To open the recent query log
        query_cache=json.load(f)

for word in new_corrected_query:                                         #Caculate the Tf-Idf score of words in query
    calc_tf_idf_score_query(word)

cosine_similarity_score(new_corrected_query)
calc_doc_length()
normalised_score()

Norm_sorted_score=sorted(Norm_score.items(), key=operator.itemgetter(1))
Norm_sorted_score.reverse()

k=min(k,len(Norm_sorted_score))                                         # If k> # of retreived docs k=# of retrieved docs
for i in range(k):
    print("DocID: "+str(Norm_sorted_score[i][1]),"Score: "+str(Norm_sorted_score[i][0]))
    print("Document Name: "+str(file_name_ID[str(Norm_sorted_score[i][0])]))
    top_k_docs.append(file_name_ID[str(Norm_sorted_score[i][0])])

temp_list=[]
query_string=' '.join(new_corrected_query)
if (len(query_cache)>=20):
     del query_cache[0]                                                 #To delete the old query        
temp_list.append(new_corrected_query)
temp_list.append(top_k_docs)
query_cache.append(temp_list)
with open('query_cache_cosine.json','w') as file:
    json.dump(query_cache,file)                                         #To store the latest query into query log