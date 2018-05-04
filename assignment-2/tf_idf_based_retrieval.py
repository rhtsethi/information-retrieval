# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 00:38:43 2018

@author: Rohit
"""
from __future__ import print_function
import sys
import json
import math
import re
from nltk.metrics import edit_distance
from num2words import num2words
from nltk.stem import PorterStemmer

with open('inverted_index.json') as f:
    inverted_index=json.load(f)
    
with open('doc_id_doc_name.json') as f:
    file_name_ID=json.load(f)

porter = PorterStemmer()    
 
#To calculate the the TF-IDF score of a query word and store in docid_score list   
def calc_tf_idf_score(word):
    posting_list=inverted_index[word]
    df=posting_list[0][0]
    for i in range(1,len(posting_list)):
        temp=[]
        temp.append(posting_list[i][0])
        temp.append((1+ math.log10(posting_list[i][1]))*(math.log10((N*1.0)/df)))
        docid_score.append(temp)


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

#To check and get the query result from Cache (if present) 
def query_caching(new_corrected_query):
    with open('query_cache.json') as f:
        query_cache=json.load(f)
    for each_query in query_cache:
        if cmp(new_corrected_query,each_query[0])==0:
            for doc in each_query[1]:
                print (str(doc))
            sys.exit()
            
query=[]                                 #query word list
new_corrected_query=[]                   #Query after processing(Spell correction,stemming,etc)
docid_score=[]                           #To store the Score of Doc-query pair
query_cache=[]                           #To store the retrieved result from query cache file
final_doc_id_score=[]                    #Score after sorting in decreasing order
k=5                                      #Number of documents retrieved                               
top_k_docs=[]                            #Top K Documents
N=len(file_name_ID)                      #Number of Documents
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


for i in range(len(query)):                              #Porter stemming on query
    query[i]=porter.stem(query[i])

for i in range(len(query)):                     #To convert number to words
   try:
       val = int(query[i])
       query[i]=num2words(val)
   except ValueError:
       continue
 
spell_correction(query)
query_caching(new_corrected_query)

with open('query_cache.json') as f:                     #To open the recent query log
        query_cache=json.load(f)
        
for word in new_corrected_query:                        #Calculate the Tf-Idf score of query_doc pair
    calc_tf_idf_score(word)

for i in range(0,len(docid_score)):                     #To combine the score of same document
    temp=[]
    temp.append(docid_score[i][0])
    total_tf_idf_score=docid_score[i][1]
    if i!=0:
        for doc_id in final_doc_id_score:
            if doc_id[0] == docid_score[i][0]:
                continue
    for j in range(i+1,len(docid_score)):
        if(docid_score[i][0]==docid_score[j][0]):
            total_tf_idf_score+=docid_score[j][1]
    temp.append(total_tf_idf_score)
    final_doc_id_score.append(temp)
    
final_doc_id_score.sort(key=lambda x:x[1],reverse=True)                 #To sort the score into descending order
    
k=min(k,len(final_doc_id_score))                                       #If k> # of retreived docs then k=# of retrieved docs
for i in range(k):
    print("DocID: "+str(final_doc_id_score[i][0]),"Score: "+str(final_doc_id_score[i][1]))
    print("Document Name: "+str(file_name_ID[str(final_doc_id_score[i][0])]))
    top_k_docs.append(file_name_ID[str(final_doc_id_score[i][0])])

temp_list=[]
query_string=' '.join(new_corrected_query)
if (len(query_cache)>=20):                                          
     del query_cache[0]                                                 #To delete the old query
temp_list.append(new_corrected_query)                                   
temp_list.append(top_k_docs)
query_cache.append(temp_list)
with open('query_cache.json','w') as file:
    json.dump(query_cache,file)                                        #To store the latest query into query log         



    

    

