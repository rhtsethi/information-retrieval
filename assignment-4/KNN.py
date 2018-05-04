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
from collections import Counter
import time


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
                        test_doc_class_list.append(directory)
                        total_test_docs_class+=1
                        continue
                    if(directory not in class_doc_name):                            
                        class_doc_name[directory]=[]
                        class_docs_words[directory]={}                              #TO store the dictionary containing the documents-words list pair of that class
                    class_doc_name[directory].append(doc_name)                      #To store the docs class wise
                    class_docs_words[directory][doc_name]=set()                        #TO store the terms document wise and documnetys class wise (dictionary of dictionary)
                    doc_class[doc_name]=directory
                    doc_id_name[i]=doc_name
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
    print(len(tf_idf_terms))
    print(docs_matrix.shape)
    for doc_id,doc_name in doc_id_name.iteritems():
        i=0
        for term,doc_name_tf_idf in tf_idf_terms.iteritems():
             if term in class_docs_words[doc_class[doc_name]][doc_name]:
                docs_matrix[doc_id][i]=doc_name_tf_idf[doc_name]
             else:
                docs_matrix[doc_id][i]=0
             i+=1

def calc_tf_idf_test_doc(term,doc):
    posting_dict=inverted_index[term]
    df=posting_dict["df"]
    return ((1+math.log10(terms_doc[term]))*(math.log10(N_docs*1.0)/df))  

  
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
        
def test_document_vector():
    j=0
    for doc in test_doc:
        terms_doc=testing_preprocess(test_doc_data[doc])
        i=0
        for term,doc_name_tf_idf in tf_idf_terms.iteritems():
             if term in terms_doc:
                test_docs_matrix[j][i]=calc_tf_idf_test_doc(term,doc)
             else:
                test_docs_matrix[j][i]=0
             i+=1  
        j+=1
    
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

def cosine_similarity(test_docs_matrix,docs_matrix):
    
    top_k_class=np.empty([test_docs_matrix.shape[0],k],dtype="S30")
    predicted_class=np.empty([test_docs_matrix.shape[0]],dtype="S30")
    norm_length=np.dot(((np.sqrt(np.sum(np.power(test_docs_matrix,2),axis=1))).reshape((test_docs_matrix.shape[0],1))),np.transpose((np.sqrt(np.sum(np.power(docs_matrix,2),axis=1))).reshape((docs_matrix.shape[0],1))))
    cos_simil=np.divide(np.dot(test_docs_matrix,docs_matrix.T),norm_length,where=norm_length!=0)
    for i in range(cos_simil.shape[0]):
        top_k_doc_id=np.argpartition(cos_simil[i], -k)[-k:]
        j=0
        for doc_id in top_k_doc_id:
            top_k_class[i][j]=(doc_class[doc_id_name[doc_id]])           
            j+=1
    
    for i in range(top_k_class.shape[0]):
        predicted,class_count = Counter(top_k_class[i]).most_common(1)[0]
        predicted_class[i]=predicted
    return predicted_class
    
print(time.strftime('%X %x %Z'))   

tokenizer= RegexpTokenizer(r'\w{3,}')
porter = PorterStemmer()
stop_words=set(stopwords.words('english'))
                                 
split_test_ratio=[0.5,0.3,0.2,0.1]              #Split ratios for test
split_ratio=["50:50","70:30","80:20","90:10"]
knn_value=[1,3,5]
accuracies=[]                                          #Best accuracy of different test split ratio at (K=1,3,5)

path='D:/MTECH/SEM 2/Information Retrieval/Assignments/Assignment 3/20_newsgroups'
for i in range(len(split_test_ratio)):
    kth_accuracy=[]
    for k in knn_value:
        N_docs=0                                        #Total # of docs
        N_docs_class=0                                  #To count the #of docs class wise
        docs_count_class={}                             #TO store the # of docs class wise
        terms=set()                                        # To store all vocabulary terms
        class_doc_name={}                               #To store the docs class wise
        doc_class={}
        inverted_index={}
        terms_doc={}
        doc_id_name={}
        tf_idf_terms={}
        class_doc_matrix={}
        class_centroid={}
        test_doc=[]                                     #To store all the test docs
        test_doc_data={}                                #To store the words as string of test docs
        test_doc_class={}                               #To store the true class of test document
        test_doc_class_list=[]
        docs_class_words={}                             #To store the words of the documents class wise
        class_id={}                                     # To store the integer id corresponding to class(for confusion matrix)
        class_docs_words={}                              #TO store the dictionary containing the documents-words list pair of that class
        extract_docs_classes(path,split_test_ratio[i])
        print("Index created")
        docs_matrix=np.zeros((len(doc_id_name),len(inverted_index)))
        test_docs_matrix=np.zeros((len(test_doc),len(inverted_index)))
        document_frequency(inverted_index)
        for term,value in inverted_index.iteritems():
            tf_idf_terms[term]={}
            for doc_name,tf in value.iteritems():
                tf_idf_terms[term][doc_name]=calc_tf_idf_score(term,doc_name)
                
        document_vector()
        print("Document vector representation completed")
        print("Training completed")
        
        correct_count=0
        confusion_arr=np.zeros((len(docs_count_class),len(docs_count_class)))           #For plotting the confusion matrix
        unique_id=0
        for key,value in docs_count_class.iteritems():                                  #For maintaining the unique id corresponding to class 
            class_id[key]=unique_id
            unique_id+=1
            
        m=0
        for doc in test_doc:
            terms_doc=testing_preprocess(test_doc_data[doc])
            l=0
            for term,doc_name_tf_idf in tf_idf_terms.iteritems():
                 if term in terms_doc:
                    test_docs_matrix[m][l]=calc_tf_idf_test_doc(term,doc)
                 else:
                    test_docs_matrix[m][l]=0
                 l+=1  
            m+=1       
        predicted_class_name=cosine_similarity(test_docs_matrix,docs_matrix)
        test_doc_class_np=np.array(test_doc_class_list)
        correct_count=np.sum(predicted_class_name==test_doc_class_np)
        for m in range(test_docs_matrix.shape[0]):
            confusion_arr[class_id[test_doc_class_np[m]]][class_id[predicted_class_name[m]]]+=1
        accuracy=(correct_count/len(test_doc))*100
        print(accuracy)
        fig,ax=plot_confusion_matrix(conf_mat=confusion_arr)
        plt.title("Confusion Matrix for "+split_ratio[i]+" Train:Test Split with K="+str(k))
        fig.savefig('KNN with K=' +str(k)+' Confusion Matrix with test split ratio '+str(split_test_ratio[i]*100)+'.png',bbox_inches='tight')
        kth_accuracy.append(accuracy)
    accuracies.append(max(kth_accuracy))
    fig=plt.figure()
    x=[a for a in range(1,len(knn_value)+1)]
    plt.plot(x,kth_accuracy)
    plt.xticks(x,knn_value)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("KNN accuracy for "+split_ratio[i]);
    fig.savefig('KNN with test split ratio '+str(split_test_ratio[i]*100)+' accuracy.png')

fig=plt.figure()
x=[a for a in range(1,len(split_ratio)+1)]
plt.plot(x,accuracies)
plt.xticks(x,split_ratio)
plt.xlabel("Train-Test Ratio")
plt.ylabel("Accuracy")
plt.title("KNN accuracy")
fig.savefig('KNN accuracy.png')
    
print(time.strftime('%X %x %Z'))