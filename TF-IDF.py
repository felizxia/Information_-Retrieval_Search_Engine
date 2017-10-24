import nltk
import string
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as py
import matplotlib.pylab as plt
import math
import operator
from nltk.tokenize import RegexpTokenizer

ehr=open('ehr.txt').read()
med=open('medhelp.txt').read()
stop_words=open('stoplist.txt','r')
stop=[]
for i in stop_words:
    stop.append(i.strip())
    
# take tokens,remove stop_words for next step's counting and remove punctuations
#### 1 way as tokenization

tokenizer = RegexpTokenizer(r'\w+')
w_p_1= tokenizer.tokenize(ehr)
w_p_2= tokenizer.tokenize(med)
e_w_p = [x for x in w_p_1 if x not in stop]
m_w_p = [x for x in w_p_2 if x not in stop]
#### 2 way as tokenization
ehr_t= nltk.word_tokenize(ehr)
ehr_g = [x for x in ehr_t if x not in stop]

med_t = nltk.word_tokenize(med)
med_g=  [x for x in med_t if x not in stop]

# plot: x-axis - word frequency (number of times a word appears in the collection)
#y-axis - proportion of words with this frequency
def ploted(corpus):
    e_dict={}
    for i in corpus:
        if i not in e_dict.keys():
            e_dict[str(i)]=1
        else:
            e_dict[str(i)]+=1
    e_count={}
    for i in e_dict.values():
        if i not in e_count:
            e_count[i]=1
        else:
            e_count[i]+=1
    list=sorted(e_count.items())
    x, y = zip(*list) 
    pplot= plt.gca()
    pplot.set_yscale('log')
    pplot.set_xscale('log')
    return pplot.scatter(x,y)
ploted(ehr_g)
plt.show()
ploted(med_g)

# stop_word_freq without moving capital letters
def stop_word_p(doc):
    stop_l= [x for x in doc if x in stop]
    percent= len(stop_l)/len(doc)
    return percent
print(stop_word_p(w_p_1))
print(stop_word_p(w_p_2))

#percentage of capital letters
def c_counter(x): 
    count=0
    c_u=0
    for i in x:
        for m in i:
            count+=1
            if m.isupper():
                c_u+=1
    return c_u/count

print('percentage of capital letters of electronic health records is')
print(c_counter(w_p_1))
print('percentage of capital letters of MedHelp textfile is')
print (c_counter(w_p_2))

#average number of characters per word
ehr_t_word_l=len(w_p_1)
med_t_word_l=len(w_p_2)
def cha_t(x):
    count=0
    for i in x:
        for m in i:
            count+=1
    return count
print (cha_t(w_p_1)/ehr_t_word_l)
print (cha_t(w_p_2)/med_t_word_l)

#percentage of nouns, adjectives, verbs, adverbs, and pronouns of ehr_t ignore punc
from collections import Counter
def auto_c(doc):
    all_e=nltk.pos_tag(doc)
    dict={}
    all_c= Counter(tag for word, tag in all_e)
    for i in all_c.items():
        dict[i[0]]= i[1]/len(doc)
    return dict
over_e=auto_c(e_w_p)
over_w=auto_c(m_w_p)

# ehr nouns(NN,NNS,NNP), adjectives(JJ JJR JJS)verbs(VBD VBG VBN VBP VBZ), adverbs(RB RBR RBS), pronouns(PRP, PRP$)
def percent(doc,type):
    n=0
    a=0
    v=0
    adv=0
    p=0
    for i,m in doc.items():
        if i in ["NN","NNS","NNP","NNPS"]:
            n+= m
        if i in ['JJ','JR','JJS']:
            a+=m
        if i in ['VBD','VBG','VBN','VBP','VBZ']:
            v+=m
        if i in ['RB','RBR','RBS']:
            adv+=m
        if i in ['PRP','PRP$']:
            p+=m
        else:
            pass
    if type=="n":
        return n
    if type=="a":
        return a
    if type=="v":
        return v
    if type=="adv":
        return adv
    if type=="p":
        return p
for i in ['n','a','v','adv','p']:
    
    print('ehr'+ '\t' +str(i)+ '\t'+ str(percent(over_e,i)))
    print('med'+ '\t' +str(i)+ '\t'+str(percent(over_w,i)))


# review all pos tag distribution
over_e
over_w

#the top 10 nouns, top 10 verbs, and top 10 adjectives of ehr_t
noun= [n for n,pos in all_e if pos in ["NN","NNS","NNP","NNPS"]]
adj = [n for n,pos in all_e if pos in ['JJ','JR','JJS']]
verb = [n for n,pos in all_e if pos in ['VBD','VBG','VBN','VBP','VBZ']]

print (nltk.FreqDist(noun).most_common(10))
print (nltk.FreqDist(adj).most_common(10))
print (nltk.FreqDist(verb).most_common(10))

# compare this to former(unformalized one)
def tokenized(file):
    tokens = tokenizer.tokenize(file.lower())
    stems_e=[]
    for word in tokens:
        stems_e.append(PorterStemmer().stem(word))
    return [word for word in stems_e if word not in stop]
#For each of the first 10 documents in the EHR collection, print out the 5 words that have the highest TF-IDF weights.
# EXTRACT 10 DOCs
ehr_test=open('ehr.txt').readlines()
ehr_doc= ehr_test[:10]

med_test= open('medhelp.txt').readlines()
med_doc=med_test[:10]
# calcuate IDF

# transform IDF

# calculate all words occurences in all of our corpus 

# calcaute N 
N_ehr= len(ehr_test)
N_med = len(med_test)
def f_IDF(doc,N):
    dict_e={}
    # docs
    doc_c=0
    for i in doc:
        doc_c+=1
        # for each docs, tokenize into list
        token= tokenized(i)
        #count each docs' words' count, but we mostly want the key: (key,1) in each docs
        dict_e[str(doc_c)]={}
        for m in token:
            if m not in dict_e[str(doc_c)].keys():
                dict_e[str(doc_c)][m]=1
            else:
                dict_e[str(doc_c)][m]+=1
    # calcualte dict's keys' occurence
    e_count={}
    for i in dict_e.keys():
        for m in dict_e[str(i)].keys():
            if m not in e_count.keys():
                e_count[str(m)]=1
            else:
                e_count[str(m)]+=1
    e_tf={}
    for x,y in e_count.items():
        e_tf[str(x)]= 1+math.log(N/y)
    return e_tf
e_IDF=f_IDF(ehr_test,N_ehr)
m_IDF=f_IDF(med_test,N_med)

# DF
# For each docs in each collection, calculate word account, which we already did in IDF, then trasform
def f_DF(doc):
    dict_e={}
    # docs
    doc_c=0
    for i in doc:
        doc_c+=1
        # for each docs, tokenize into list
        token= tokenized(i)
        #count each docs' words' count, but we mostly want the key: (key,1) in each docs
        dict_e[str(doc_c)]={}
        for m in token:
            if m not in dict_e[str(doc_c)].keys():
                dict_e[str(doc_c)][m]=1
            else:
                dict_e[str(doc_c)][m]+=1
# transform df to log
    df_tf={}
    for i in dict_e.keys():
        df_tf[i]={}
        for x,y in dict_e[i].items():
            df_tf[i][x]= math.log(y+1)
    return df_tf
# remeber we only want 10 doc's item frequency
e_df = f_DF(ehr_doc)
f_df = f_DF(med_doc)

# for each doc 1-10, calculate the df-idf
e_df_idf={}

for i in e_df.keys():
    e_df_idf[i]= dict()
    for d in e_df[i].keys(): 
         e_df_idf[i][d] = e_df[i][d] * e_IDF[d]
e_df_idf

for i in e_df_idf.keys():
    print (i)
    list=sorted(e_df_idf[i].items(),key=lambda x:x[1],reverse=True)[:5]
    lists=[i[0] for i in list]
    print(lists)
f_df_idf={}

for i in f_df.keys():
    f_df_idf[i]= dict()
    for d in f_df[i].keys(): 
         f_df_idf[i][d] = f_df[i][d] * m_IDF[d]
f_df_idf

for i in f_df_idf.keys():
    print (i)
    listf=sorted(f_df_idf[i].items(),key=lambda x:x[1],reverse=True)[:5]
    listsf=[i[0] for i in listf]
    print(listsf)

# calculate DF * IDF

e_DF_IDF = {e: (e_DF[e] * e_IDF[e]) for e in e_DF}
e_top5= sorted(e_DF_IDF.items(),key=lambda x:x[1],reverse=True)[:5]
m_DF_IDF = {m: (m_DF[e] * m_IDF[e]) for m in m_DF}
m_top5= sorted(m_DF_IDF.items(),key=lambda x:x[1],reverse=True)[:5]

e_top5

m_top5
