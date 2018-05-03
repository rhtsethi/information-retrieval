# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 08:57:37 2018

@author: Rohit
"""
from __future__ import print_function
"""This File takes the query as command line argument and performs the merge/union/and_not/or_not operation.
"""

import json
import sys
from nltk.stem import PorterStemmer
import math
from timeit import default_timer as timer
import matplotlib.pyplot as plt



porter = PorterStemmer()

with open('inverted_index.json') as f:
    inverted_index=json.load(f)
    
with open('doc_id_doc_name.json') as f:
    file_name_ID=json.load(f)


def merge_intersect(posting_first,posting_second):
    i=0
    j=0
    start=timer()
    intersect=[]
    while (i!=len(posting_first) and j!=len(posting_second)):
        if posting_first[i]==posting_second[j]:
            intersect.append(posting_first[i])
            i+=1
            j+=1
        elif posting_first[i]<posting_second[j]:
            i+=1
        else:
            j+=1
    print('Time for Intersect Using Merge Algortihm :',timer()-start)
    return intersect

def merge_union(posting_first,posting_second):
    i=0
    j=0
    union=[]
    while (i!=len(posting_first) and j!=len(posting_second)):
        if posting_first[i]==posting_second[j]:
            union.append(posting_first[i])
            i+=1
            j+=1
        elif posting_first[i]<posting_second[j]:
            union.append(posting_first[i])
            i+=1
        else:
            union.append(posting_second[j])
            j+=1
    while (i!=len(posting_first)):
        union.append(posting_first[i])
        i+=1
    while (j!=len(posting_second)):
        union.append(posting_second[j])
        j+=1
    return union

def merge_and_not(posting_first,posting_second):
    i=0
    j=0
    and_or=[]
    while (i!=len(posting_first) and j!=len(posting_second)):  
        if posting_first[i]<posting_second[j]:
            and_or.append(posting_first[i])
            i+=1
        elif posting_first[i]==posting_second[j]:
            i+=1
            j+=1
        else:
            j+=1
    while i!=len(posting_first):
        and_or.append(posting_first[i])
        i+=1
    return and_or
 
def merge_or_not(posting_first,posting_second):
    all_doc_ID=[]
    for key in file_name_ID.keys():
        all_doc_ID.append(int(key))
    all_doc_ID.sort()
    temp_doc_ID=merge_and_not(all_doc_ID,posting_second)
    union_all_first=merge_union(temp_doc_ID,posting_first)  
    return union_all_first
        
def skip_posting(posting_first,posting_second):
    posting_first_tuple=[]
    posting_second_tuple=[]
    for i in posting_first:
        x=[i,-1]
        posting_first_tuple.append(x)
    for i in posting_second:
        x=[i,-1]
        posting_second_tuple.append(x)
    
    number_of_skips_first_list=[int((math.sqrt(len(posting_first_tuple)))/3),int((math.sqrt(len(posting_first_tuple)))/2),int(math.sqrt(len(posting_first_tuple))),2*int(math.sqrt(len(posting_first_tuple))),3*int(math.sqrt(len(posting_first_tuple))),4*int(math.sqrt(len(posting_first_tuple)))]
    number_of_skips_second_list=[int((math.sqrt(len(posting_second_tuple)))/3),int((math.sqrt(len(posting_second_tuple)))/2),int(math.sqrt(len(posting_second_tuple))),2*int(math.sqrt(len(posting_second_tuple))),3*int(math.sqrt(len(posting_second_tuple))),4*int(math.sqrt(len(posting_second_tuple)))]
    number_of_skips_first_list.sort()
    number_of_skips_second_list.sort()
    time_list=[]
    comparison_list=[]
    for p in range(len(number_of_skips_first_list)):     
        number_of_skips_first=number_of_skips_first_list[p]
        number_of_skips_second=number_of_skips_second_list[p]
        
        i=0;count=0
        while (i<=len(posting_first_tuple)):
            k=int(((len(posting_first))/number_of_skips_first)*(count+1))
            if k<len(posting_first_tuple) and k>0:
                posting_first_tuple[i][1]=k
            count+=1
            i=k
        i=0;count=0
        while (i<=len(posting_second_tuple)):
            k=int(((len(posting_second))/number_of_skips_second)*(count+1))
            if k<len(posting_second_tuple) and k>0:
                posting_second_tuple[i][1]=k
            count+=1
            i=k  
        start=timer()     
        intersect=[]
        comparison=0
        i=0;j=0
        while(i!=len(posting_first_tuple) and j!=len(posting_second_tuple)):
            if posting_first_tuple[i][0]==posting_second_tuple[j][0]:
                comparison+=1
                intersect.append(posting_first_tuple[i][0])
                i+=1
                j+=1
            elif posting_first_tuple[i][0]<posting_second_tuple[j][0]:
                comparison+=1
                if posting_first_tuple[i][1]!=-1 and (posting_first_tuple[posting_first_tuple[i][1]][0]<=posting_second_tuple[j][0]):
                    while  posting_first_tuple[i][1]!=-1 and (posting_first_tuple[posting_first_tuple[i][1]][0]<=posting_second_tuple[j][0]):
                        comparison+=1
                        i=posting_first_tuple[i][1]
                else:
                    i+=1
            else:
                comparison+=1
                if posting_second_tuple[j][1]!=-1 and (posting_second_tuple[posting_second_tuple[j][1]][0]<=posting_first_tuple[i][0]):
                    while posting_second_tuple[j][1]!=-1 and (posting_second_tuple[posting_second_tuple[j][1]][0]<=posting_first_tuple[i][0]):
                        comparison+=1
                        j=posting_second_tuple[j][1]
                else:
                    j+=1
        print("Number of skips for first posting list:",number_of_skips_first_list[p])
        print("Number of skips for second posting list:",number_of_skips_second_list[p])
        total_time=timer()-start
        print("Time taken:",total_time)
        print("Comparisons performed:",comparison)
        time_list.append(total_time)
        comparison_list.append(comparison)
        print("Number of documents retrieved: ",len(intersect))
        print()
    fig=plt.figure()
    title='Query: '+sys.argv[1]+' '+sys.argv[2]+' '+sys.argv[3]
    plt.title(title)
    plt.plot(number_of_skips_first_list,time_list)
    plt.xlabel("Number of Skips (First List)")
    plt.ylabel("Time(in second)")
    fig.savefig('skips_vs_time.png')
    fig=plt.figure()
    title='Query: '+sys.argv[1]+' '+sys.argv[2]+' '+sys.argv[3]
    plt.title(title)
    plt.plot(number_of_skips_first_list,comparison_list)
    plt.xlabel("Number of Skips (First List)")
    plt.ylabel("Number of Comparisons")
    fig.savefig('skips_vs_comparison.png')
    return intersect
    
    
def display(list_display):
    for i in list_display:
        i=str(i)
        print(file_name_ID[i],end=', ')

    
if (len(sys.argv)==4):
    first_word=sys.argv[1]
    operator=sys.argv[2]
    second_word=sys.argv[3] 
    print('Query: ',first_word+" "+operator+" "+second_word)
    first_word=porter.stem(first_word)
    second_word=porter.stem(second_word)
    posting_first=inverted_index[first_word]
    posting_second=inverted_index[second_word]
    print('Length of first word posting list: ',len(posting_first))
    print('Length of second word posting list: ',len(posting_second))
    print()
    if operator=='AND' or operator=='and':
        intersect_list=merge_intersect(posting_first,posting_second)
        print('Number of documents retrieved using merge algorithm: ',len(intersect_list))
        display(intersect_list)
        print()
        print()
        intersect_skip_list=skip_posting(posting_first,posting_second)
        print()
        print('Number of documents retrieved using skip pointer algorithm: ',len(intersect_skip_list))
        display(intersect_skip_list)
    elif operator=='OR' or operator=='or':
        union_list=merge_union(posting_first,posting_second)
        print('Number of documents retrieved: ',len(union_list))
        display(union_list)
        
if (len(sys.argv)==5):
    first_word=sys.argv[1]
    operator_one=sys.argv[2]
    operator_two=sys.argv[3]
    second_word=sys.argv[4]
    print('Query: ',first_word+" "+operator_one+" "+operator_two+" "+second_word)
    first_word=porter.stem(first_word)
    second_word=porter.stem(second_word)   
    posting_first=inverted_index[first_word]
    posting_second=inverted_index[second_word]
    print('Length of first word posting list: ',len(posting_first))
    print('Length of second word posting list: ',len(posting_second))
    if operator_one=='AND' or operator_one=='and':
        if operator_two=='NOT' or operator_two=='not':
            and_not=merge_and_not(posting_first,posting_second)
            print ('Number of documents retrieved: ',len(and_not))
            display(and_not)
    if operator_one=='OR' or operator_one=='or':
        if operator_two=='NOT' or operator_two=='not':
            or_not=merge_or_not(posting_first,posting_second)
            print('Number of documents retrieved: ',len(or_not))
            display(or_not)


    

        
    



        
        
        
        
        
    