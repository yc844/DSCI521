# -*- coding: utf-8 -*-
"""
# C:\DSCI521\project
"""
#%%
import pandas as pd
tw1 = pd.read_csv(r"C:\DSCI521\project\train.csv")

import spacy 
nlp = spacy.load('en_core_web_sm')


def nlp_text(text) :
    s = ''
    #make a spacy nlp object 
    t_nlp = nlp(text)
    for token in t_nlp:
        s = s + token.text+' '+token.pos_+' ' +  token.tag_ + ' ' +token.dep_+ ' '
#    print('s=',s )
    return s

#little test of the above function using twitter row data 
# t5 = 'Accident in #Ashville on US 23 SB before SR 752 #traffic http://t.co/hylMo0WgFI'
# x = nlp_text(t5)
# print(x)

#apply the function to the text column and create new column nlp_col 
tw1['nlp_col']= tw1['text'].apply(nlp_text) 

#rearrange the columns for X and y 
# Re-order Columns text and nlp_col 
#tw1  = tw1[['id','keyword','text','nlp_col','target']]

#recorder and cut 'text column
#tw1  = tw1[['id','keyword','nlp_col','target']]
tw1  = tw1[['nlp_col','target']]

#create X and y data 
X = tw1['nlp_col']
y = tw1['target']

#%%


#y = y[:-1]  #odd number of rows or test size 
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .10 , random_state = 0)

#%%


#vectorize for ML 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Initialize count vectorizer
count_vectorizer = CountVectorizer(min_df=0.05, max_df=0.9)
#%%
# Create count train and test variables
count_train = count_vectorizer.fit_transform(X_train, y_train)
count_test = count_vectorizer.transform(X_test)
#%%
# Initialize tfidf vectorizer
#tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=0.05, max_df=0.9)
tfidf_vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.9)


#%%


# Create tfidf train and test variables
tfidf_train = tfidf_vectorizer.fit_transform(X_train, y_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
#%%

# Create a MulitnomialNB model
print("model created")
tfidf_nb = MultinomialNB()
#%%
# ... Train your model here ...
#error Found input variables with inconsistent numbers of samples: [2, 6850]
#model = forest.fit(train_fold, train_y.values.ravel())

print("model trained")
'''
760: DataConversionWarning: A column-vector y was passed when a 1d array was expected.
 Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
https://stackoverflow.com/questions/15115765/how-to-access-sparse-matrix-elements

'''
tfidf_nb.fit(tfidf_train, y_train)
#%%
# Run predict on your TF-IDF test data to get your predictions
tfidf_nb_pred = tfidf_nb.predict(tfidf_test)

# Calculate the accuracy of your predictions
tfidf_nb_score = metrics.accuracy_score(tfidf_nb_pred, y_test)

# Create a MulitnomialNB model

count_nb = MultinomialNB()

# ... Train your model here ...
count_nb.fit(count_train, y_train)

# Run predict on your count test data to get your predictions
count_nb_pred = count_nb.predict(count_test)

# Calculate the accuracy of your predictions
count_nb_score = metrics.accuracy_score(count_nb_pred, y_test)

print('NaiveBayes Tfidf Score: ', tfidf_nb_score)
print('NaiveBayes Count Score: ', count_nb_score)