# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:41:29 2018

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

#To split the docs into train and test and to perform preprocessing and store the train docs class wise
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
                    class_doc_name[directory].append(doc_name)                      #To store the docs class wise
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
                                #inverted_index[stemmed_word]=[]
                                terms.add(stemmed_word)                                 #To include all vocabulary terms in 'terms' set
                            if(directory not in docs_class_words):
                                docs_class_words[directory]=[]
                            docs_class_words[directory].append(stemmed_word)        #To include all the words of the documents class wise
                    i+=1
                    N_docs_class+=1
                if(directory not in docs_count_class):
                    docs_count_class[directory]=N_docs_class                        #To store the number of docs class wise
                N_docs+=N_docs_class                                                #Total # of documents
                N_docs_class=0
                
#    print(docs_count_class)
#    print(class_doc_name['sci.space'])
#    print(N_docs)

#Training of the train set using Naive Bayes Method
def trainingNB(docs_count_class,class_doc_name):
    global N_docs
    global prior
    global docs_class_words
    global conditional_prob
    for key,values1 in docs_count_class.iteritems():
        prior[key]=docs_count_class[key]/N_docs
        count_terms={}                                        #To store the count of each term in that class
#        for term in terms:
#            count=0
#            for word in docs_class_words[key]:
#                if(word==term):
#                    count+=1
#            count_terms[str(term)]=count
#            
        for word in docs_class_words[key]:
            if word in terms:
                if(word not in count_terms):
                    count_terms[word]=0
                count_terms[word]+=1                        #Count of each term in that class is updated
        
        if(key not in conditional_prob):
            conditional_prob[key]={}
#        for term,values2 in count_terms.iteritems():
#            conditional_prob[key][term]=(((count_terms[term]+1)*1.0)/((sum(count_terms.values())+len(terms))*1.0))
         
        for term in terms:
            if(term not in count_terms):                    #Terms that doesn't belong to that class
                conditional_prob[key][term]=(((1)*1.0)/((sum(count_terms.values())+len(terms))*1.0))
            else:
              conditional_prob[key][term]=(((count_terms[term]+1)*1.0)/((sum(count_terms.values())+len(terms))*1.0))    #Conditional prob stores the term-class pair wise conditional probability

#For preprocessing of test document words         
def testing_preprocess(string_of_words):
     tokens_doc=[]
     words=tokenizer.tokenize(string_of_words)               # To remove punctuation mark, comma etc and to form tokens
     for word in words:
         if (word not in stop_words):                                # Removing Stop Words
             word=word.lower()                                       # Normalization
             try:
                 stemmed_word=porter.stem(word)
                 stemmed_word = unicode(stemmed_word, errors='ignore')
             except:
                 pass
             tokens_doc.append(stemmed_word)
     return tokens_doc

#To test a document(defined by string_of_words) for predictng the class using Naive Bayes method             
def testingNB(string_of_words):
    global prior
    global conditional_prob
    score={}                                                    #Score of each class is maintained (to predict the class with maximum score)
    tokens=testing_preprocess(string_of_words)                  #To get the processed 'tokens' of test document
    for class_key,values in conditional_prob.iteritems():
        score[class_key]=math.log(prior[class_key]*1.0)         #score is updated with prior of that class
        for token in tokens:
            if(token in conditional_prob[class_key]):                   #To include only those tokens present in vocabulary
                score[class_key]+=math.log(conditional_prob[class_key][str(token)]*1.0)     #score is updated with conditional probability of that class-token pair
    return (max(score, key=score.get))                          #Return the class with maximum score


    


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
    prior={}                                        #To store the prior of each class
    docs_class_words={}                             #To store the words of the documents class wise
    conditional_prob={}                             #TO store the conditional probability of class and term pair(It is dictionary of dictionary)
    test_doc=[]                                     #To store all the test docs
    test_doc_data={}                                #To store the words as string of test docs
    test_doc_class={}                               #To store the true class of test document
    class_id={}                                     # To store the integer id corresponding to class(for confusion matrix)
    extract_docs_classes(path,split_test_ratio[i])
    print(docs_count_class)
    trainingNB(docs_count_class,class_doc_name)
    print("Training completed")
    
    correct_count=0
    confusion_arr=np.zeros((len(docs_count_class),len(docs_count_class)))           #For plotting the confusion matrix
    unique_id=0
    for key,value in docs_count_class.iteritems():                                  #For maintaining the unique id corresponding to class 
        class_id[key]=unique_id
        unique_id+=1
    
    for doc in test_doc:                                                            
        predicted_class=testingNB(test_doc_data[doc])                               #Predicting the class
        if predicted_class==(test_doc_class[doc]):
            correct_count+=1
        confusion_arr[class_id[test_doc_class[doc]]][class_id[predicted_class]]+=1
    accuracy=(correct_count/len(test_doc))*100
    print (accuracy)
    accuracies.append(accuracy)
    fig,ax=plot_confusion_matrix(conf_mat=confusion_arr)
    plt.title("Confusion Matrix for "+split_ratio[i]+" Train:Test Split")
    fig.savefig('Confusion Matrix with test split ratio '+str(split_test_ratio[i]*100)+'.png',bbox_inches='tight')

fig=plt.figure()
x=[a for a in range(1,len(split_ratio)+1)]
plt.plot(x,accuracies)
plt.xticks(x,split_ratio)
plt.xlabel("Train-Test Ratio")
plt.ylabel("Accuracy")
fig.savefig('accuracy.png')
    







