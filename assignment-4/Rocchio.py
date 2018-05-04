# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 16:38:28 2018

@author: Rohit
"""

from __future__ import print_function
from __future__ import division
import os
import math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from random import shuffle
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
from scipy import sparse


#To remove the Text before the file Blank Line in all the Documents
def header_preprocess(input_path_file):
    lines=open(input_path_file,'r').readlines()
    for i,line in enumerate(lines):
        line=line.strip()
        if not line:
            break
    input_file_list= open(input_path_file,'r').readlines()[i+1:]
    complete_file_string=''.join(input_file_list)
    return complete_file_string

def extract_docs_classes(path,split_ratio):
    i=0
    global N_docs_class
    global N_docs
    global class_doc_name
    global docs_count_class
    global terms
    global docs_class_words
    global test_doc
    global test_doc_data
    global test_doc_class
    global docs_name_words
    global inverted_index
    for root,dirs,files in os.walk(path,topdown=False): 
                number_of_docs=len(os.listdir(root))
                total_test_docs_class=0                            #Total docs included in training set of each class
                files_list=files
                shuffle(files_list)                                 # Shuffling the docs of one class(for random train-test split)
                for name in files_list:
                    directory=root.split("\\",1)[1]
                    doc_name=directory+'/'+name
                    path_file=os.path.join(root,name)
                    header_processed_string=header_preprocess(path_file)        #For header preprocessing
                    if(total_test_docs_class<(split_ratio*number_of_docs)):                   #To include the first k docs into test set
                        #add that doc to test set
                        test_doc.append(doc_name)
                        test_doc_data[doc_name]=header_processed_string
                        test_doc_class[doc_name]=directory
                        total_test_docs_class+=1
                        continue
                    if(directory not in class_doc_name):                            
                        class_doc_name[directory]=[]
                        class_docs_words[directory]={}                              #TO store the dictionary containing the documents-words list pair of that class
                    class_doc_name[directory].append(doc_name)                      #To store the docs class wise
                    class_docs_words[directory][doc_name]=set()                        #TO store the terms document wise and documnetys class wise (dictionary of dictionary)
                    words=tokenizer.tokenize(header_processed_string)               # To remove punctuation mark, comma etc and to form tokens
                    for word in words:
                        if (word not in stop_words):                                # Removing Stop Words
                            word=word.lower()                                       # Normalization
                            try:
                                stemmed_word=porter.stem(word)                      #Porter stemming on words
                                stemmed_word = unicode(stemmed_word, errors='ignore')
                            except:
                                pass
                            if (stemmed_word not in terms):
                                terms.add(stemmed_word)                                 #To include all vocabulary terms in 'terms' set
                            if (stemmed_word not in inverted_index):
                                inverted_index[stemmed_word]={}
                            if ((len(inverted_index[stemmed_word])==0) or (doc_name not in inverted_index[stemmed_word])):
                                inverted_index[stemmed_word][doc_name]=1
                            else:
                                inverted_index[stemmed_word][doc_name]+=1
                            if(directory not in docs_class_words):
                                docs_class_words[directory]=[]
                            class_docs_words[directory][doc_name].add(stemmed_word)          #To include the terms document wise
                            docs_class_words[directory].append(stemmed_word)        #To include all the words of the documents class wise
                    i+=1
                    N_docs_class+=1
                if(directory not in docs_count_class):
                    docs_count_class[directory]=N_docs_class                        #To store the number of docs class wise
                N_docs+=N_docs_class                                                #Total # of documents
                N_docs_class=0

def document_frequency(inverted_index):
    for key,value in inverted_index.iteritems():
        inverted_index[key]["df"]=len(value)
        
def calc_tf_idf_score(word,doc_name):
    posting_dict=inverted_index[word]
    df=posting_dict["df"]
    return ((1+math.log10(posting_dict[doc_name]))*(math.log10(N_docs*1.0)/df))
        
        
def document_vector():
    for class_name,docs in class_doc_name.iteritems():
        class_doc_vector=np.zeros((len(docs),len(inverted_index)))
        print(len(tf_idf_terms))
        print(class_doc_vector.shape)
        j=0
        for term,doc_name_tf_idf in tf_idf_terms.iteritems():
            i=0
            for doc_name,words in class_docs_words[class_name].iteritems():
                if term in words:
                    class_doc_vector[i][j]=doc_name_tf_idf[doc_name]
                else:
                    class_doc_vector[i][j]=0
                i+=1
            j+=1
        class_doc_sparse=sparse.csr_matrix(class_doc_vector)
        class_doc_matrix[class_name]=class_doc_sparse
  
def train_rocchio():
    for class_name,doc_matrix in class_doc_matrix.iteritems():
        class_centroid[class_name]=doc_matrix.sum(axis=0)/(len(class_doc_name[class_name]))

def testing_preprocess(string_of_words):
     terms_doc={}
     words=tokenizer.tokenize(string_of_words)               # To remove punctuation mark, comma etc and to form tokens
     for word in words:
         if (word not in stop_words):                                # Removing Stop Words
             word=word.lower()                                       # Normalization
             try:
                 stemmed_word=porter.stem(word)
                 stemmed_word = unicode(stemmed_word, errors='ignore')
             except:
                 pass
             if(stemmed_word in terms_doc):
                 terms_doc[stemmed_word]+=1
             else:
                 terms_doc[stemmed_word]=1
     return terms_doc

def euclidean_distance(doc,class_centroid):
    min_dist=999999
    for class_name,centroid in class_centroid.iteritems(): 
        temp_np=doc-class_centroid[class_name]
        dist=np.sum(np.power(temp_np,2))
        eucl_dist=math.sqrt(dist)
        if(min_dist>eucl_dist):
            min_dist=eucl_dist
            class_doc=class_name
    return class_doc

def cosine_similarity(doc,class_centroid):
    cos_sim=-999999
    for class_name,centroid in class_centroid.iteritems(): 
         norm_length=((np.sqrt(np.sum(np.power(doc,2))))*(np.sqrt(np.sum(np.power(class_centroid[class_name],2)))))
         if(norm_length!=0):
             cos_simil=(np.sum(np.multiply(doc,class_centroid[class_name])))/norm_length
         else:
             cos_simil=0
         if(cos_simil>cos_sim):
             cos_sim=cos_simil
             class_doc=class_name
    return class_doc

def calc_tf_idf_test_doc(term,doc):
    return ((1+math.log10(terms_doc[term]))*(math.log10(N_docs*1.0)/1))
    

tokenizer= RegexpTokenizer(r'\w{3,}')
porter = PorterStemmer()
stop_words=set(stopwords.words('english'))
                                 
split_test_ratio=[0.5,0.3,0.2,0.1]              #Split ratios for test
split_ratio=["50:50","70:30","80:20","90:10"]

accuracies=[]

path='D:/MTECH/SEM 2/Information Retrieval/Assignments/Assignment 3/20_newsgroups'
for i in range(len(split_test_ratio)):
    N_docs=0                                        #Total # of docs
    N_docs_class=0                                  #To count the #of docs class wise
    docs_count_class={}                             #TO store the # of docs class wise
    terms=set()                                        # To store all vocabulary terms
    class_doc_name={}                               #To store the docs class wise
    inverted_index={}
    tf_idf_terms={}
    class_doc_matrix={}
    class_centroid={}
    test_doc=[]                                     #To store all the test docs
    test_doc_data={}                                #To store the words as string of test docs
    test_doc_class={}                               #To store the true class of test document
    docs_class_words={}                             #To store the words of the documents class wise
    class_id={}                                     # To store the integer id corresponding to class(for confusion matrix)
    class_docs_words={}                              #TO store the dictionary containing the documents-words list pair of that class
    extract_docs_classes(path,split_test_ratio[i])
    print("Index created")
    document_frequency(inverted_index)
    for term,value in inverted_index.iteritems():
        tf_idf_terms[term]={}
        for doc_name,tf in value.iteritems():
            tf_idf_terms[term][doc_name]=calc_tf_idf_score(term,doc_name)
            
    document_vector()
    print("Document vector representation completed")
    train_rocchio()
    print("Training completed")
    
    correct_count=0
    confusion_arr=np.zeros((len(docs_count_class),len(docs_count_class)))           #For plotting the confusion matrix
    unique_id=0
    for key,value in docs_count_class.iteritems():                                  #For maintaining the unique id corresponding to class 
        class_id[key]=unique_id
        unique_id+=1
        
    for doc in test_doc:
        j=0
        doc_vector=np.zeros((1,len(inverted_index)))
        terms_doc=testing_preprocess(test_doc_data[doc])
        for term,doc_name_tf_idf in tf_idf_terms.iteritems():
            if term in terms_doc:
                    doc_vector[0][j]=calc_tf_idf_test_doc(term,doc)
            else:
                    doc_vector[0][j]=0
            j+=1
    #    predicted_class_name=euclidean_distance(doc_vector,class_centroid)
        predicted_class_name=cosine_similarity(doc_vector,class_centroid)
        if predicted_class_name==(test_doc_class[doc]):
            correct_count+=1
        confusion_arr[class_id[test_doc_class[doc]]][class_id[predicted_class_name]]+=1
    accuracy=(correct_count/len(test_doc))*100
    print(accuracy)
    accuracies.append(accuracy)
    fig,ax=plot_confusion_matrix(conf_mat=confusion_arr)
    plt.title("Confusion Matrix for "+split_ratio[i]+" Train:Test Split")
    fig.savefig('Rocchio Confusion Matrix with test split ratio '+str(split_test_ratio[i]*100)+'.png',bbox_inches='tight')

fig=plt.figure()
x=[a for a in range(1,len(split_ratio)+1)]
plt.plot(x,accuracies)
plt.xticks(x,split_ratio)
plt.xlabel("Train-Test Ratio")
plt.ylabel("Accuracy")
plt.title("Rocchio Accuracy")
fig.savefig('Rocchio accuracy.png')
    
