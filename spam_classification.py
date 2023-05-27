#importing all the necessary libraries
import glob
import os
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

#check the folder properties for the files formats in the folder
#then use the glob function to iterate through our folder for a certain file format
df = glob.glob('C:\mycode\AegonMasters\enron1\*\*.txt')
ham = glob.glob('C:\mycode\AegonMasters\enron1\ham\*.txt')
spam = glob.glob('C:\mycode\AegonMasters\enron1\spam\*.txt')
spamm = []
hamm = []

#for each message in spam folder, read all and return it on the same line
for i in spam:
    try:
        f = open(i, 'r')
        spamm.append(' '.join(f.read().splitlines()))
    except:
        None
    continue

#turn the appended data into dataframe, it will be repeated for the ham too
spamm = pd.DataFrame(spamm, columns=['spam_ham'])


#for each message in ham folder, read all and return it on the same line
for i in ham[:]:
    g = open(i, 'r') #open i- which is the file and 'r'- read the content
    hamm.append(' '.join(g.read().splitlines())) #append each file
hamm = pd.DataFrame(hamm, columns=['spam_ham'])


#creating a new column in the pandas dataframe named result and the spam was labeled as 1
spamm['result'] = spamm.apply(lambda x: 1 for x in spamm)

#creating a new column in the pandas dataframe named result and the ham was labeled as 0
hamm['result'] = hamm.apply(lambda x: 0 for x in hamm)
#this is the first method in cleaning the messages
'''
#creating a new column in the pandas dataframe named result and the spam was labeled as 1
spamm['result'] = spamm.apply(lambda x: 1 for x in spamm)

#creating a new column in the pandas dataframe named result and the ham was labeled as 0
hamm['result'] = hamm.apply(lambda x: 0 for x in hamm)

#concatenating spamm and hamm on the common column
df = pd.concat([spamm,hamm], ignore_index=True)

#print(df[df['result']==1])
#print(df[df['result']==0])

#turning every word to lowercase for uniformity
df['spam_ham'] = df['spam_ham'].apply(lambda x: x.lower())

#removing everything that isnt a number or a letter and replacing it with a space
df['spam_ham'] = df['spam_ham'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x))

#tokenization of each row
df['spam_ham'] = df['spam_ham'].apply(lambda x: word_tokenize(x))


s_words = stopwords.words('english')
#for each word in x which is the whole row, remove the stopwords
df['spam_ham'] = df['spam_ham'].apply(lambda x: [i for i in x if i not in set(s_words)])


lemma = WordNetLemmatizer()
df['spam_ham'] = df['spam_ham'].apply(lambda x: ' '.join([lemma.lemmatize(i) for i in x]))
df['spam_ham'] = df['spam_ham'].apply(lambda x: x.replace('subject', ''))
#i printed this out because i saw something relating to the government in it, to confirm if it is a spam message or not
#print(df['spam_ham'][3])
print(df)'''

#this is the second method 
spamm_list = []
hamm_list = []

s_words = stopwords.words('english')
lemma = WordNetLemmatizer()

#a function containing every needed preprocessing methods
def text_preprocessing():
    for x in spamm['spam_ham']:
        #removing everything that isnt a number or a letter and replacing it with a space
        result = re.sub('[^a-zA-Z0-9]', ' ', x)
        #words tokenization of each row
        result = word_tokenize(result)
        #for each word in x which is the whole row, remove the stopwords
        result = ' '.join([lemma.lemmatize(i) for i in result if i not in set(s_words)])
        #turning every word to lowercase for uniformity
        result = result.lower()
        #i removed the string subject before each message
        result = result.replace('subject', '')
        #then appended to the right empty list
        spamm_list.append(result)
        
    for x in hamm['spam_ham']:
        result = re.sub('[^a-zA-Z0-9]', ' ', x)
        result = word_tokenize(result)
        result = ' '.join([lemma.lemmatize(i) for i in result if i not in set(s_words)])
        result = result.lower()
        result = result.replace('subject', '')
        hamm_list.append(result)
            
#called back the function
text_preprocessing()
#created a new dataframe of the result
cleaned_spam = pd.DataFrame({'spam_ham':spamm_list})
#then i replaced the spamm dataframe 'spam_ham' column with cleaned_spam
spamm['spam_ham'] = cleaned_spam

cleaned_ham = pd.DataFrame({'spam_ham':hamm_list})
hamm['spam_ham'] = cleaned_ham
#print(spamm)
#print(hamm)

#then i concatenated the two of them
df = pd.concat([spamm,hamm], ignore_index=True)

#cleaned_df = pd.DataFrame({'spam_ham':yam})

#tfidf vectorizer helps pass text into a model by converting the words/letters into values which the model understands
tfidf = TfidfVectorizer()
tfidf.fit(df['spam_ham'])
tfidf_result = tfidf.transform(df['spam_ham']).toarray()

#then i passed the vectorized spam_ham column into X and the result as y
X = tfidf_result
y = df['result']


#split the data into test and train set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

#model
lr = LogisticRegression(C=3)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

#to confirm that my model is not overfitted and will perform well on training and test dataset
#I got an accuracy of 98% - this means my model will perform well on a message it has not seen before and classify it almost correctly, whether it is a spam or ham message
print(lr.score(X_test,y_test))

#the classification report metrics showing the f1-score and the accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
<<<<<<< HEAD

#message = input('the message? ')
#data = tfidf.transform([message]).toarray()
#result = (lr.predict(data))
#for i in result:
#    if i == '1':
#        print('spam')
#    else:
#        print('ham')
        

st.title('Spam or Ham Classifier')
def lol():
    user = st.text_area('enter any news/message/mail')
    if len(user) < 1:
        st.write(' ')
    else:
        message = user
        data = tfidf.transform([message]).toarray()
        result = (lr.predict(data))
        if result == [1]:
            print('spam')
        else:
            print('ham')
lol()
=======
>>>>>>> ded95f225bf2358f91dd9ff0bc504404840286aa
