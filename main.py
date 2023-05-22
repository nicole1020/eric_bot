"""Computer Science Capstone C964 | Nicole Mau | nmau@wgu.edu | 001336361 | eric_bot | email response in corporations"""
import csv
import logging
import mmap
import pickle
from datetime import timedelta

import sklearn
import smart_open
import vectorize as vectorize
from imblearn.over_sampling import SMOTE
from sklearn import metrics, tree
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import os
import gensim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import sqlite3 as sql
from sklearn.metrics import classification_report
import collections
import random
from tkinter import *
import tkinter.ttk as ttk

from email_helper import send_message

logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)
# I want a version of this email every x amount of time. Will use this function timer
# https://realpython.com/python-timer/
# if customer placed order recently (similar items)
# if customer has not placed order recently
# if customers have sent similar emails - send escalation email or notice out.


print('this email here:')
send_message("eric.capstone.api@gmail.com", "eric.capstone.api@gmail.com", "hello from eric capstone", "test message 1",
             user_id='me')

# create sqllite sql database for future sensitive customer data live/practical application would have a separate
# server for customer data
# db corrupted somehow -renamed and recreated db file-
connect_database = sql.connect('my_test_db.db')


###############################################################################
# create table in db running into issues with .execute command
# https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-execute.html


# connect to local database create table in db running into issues with .execute command
# https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-execute.html
# https://stackoverflow.com/questions/53128279/how-to-print-output-from-sqlite3-in-python
#


def clear_customer_data():
    try:
        delete_all = "DROP TABLE CUSTOMER"
        connect_database.execute(delete_all)
    except ValueError as e:
        print('An error occurred: %s' % e)


# clear_customer_data()


def clear_order_data():
    try:
        delete_all = "DELETE FROM ORDERS WHERE id IN (11, 12, 13, 14, 15, 16 ,17, 18,19);"
        connect_database.execute(delete_all)
    except ValueError as e:
        print('An error occurred: %s' % e)


# clear_order_data()


###############################################################################

def create_customer_table():
    try:
        connect_database.execute(''' 
        CREATE TABLE IF NOT EXISTS customer ( 
        id INTEGER  PRIMARY KEY AUTOINCREMENT,
               name TEXT,
              email TEXT )
      ''')
    except ValueError as e:
        print('An error occurred: %s' % e)


create_customer_table()


def add_customer_data():
    try:
        sql = 'INSERT INTO customer(name, email) values(?,?)'
        customer_table = [('John Doe', 'jdoe@email.com'),
                          ('Johnny Doe', 'jhdoe@email.com'),
                          ('Jane Doe', 'jedoe@email.com'),
                          ('Janey Do', 'jydoe@email.com'),
                          ('Joey Doe', 'jodoe@email.com'),
                          ('Katsu Dog', 'kdog@email.com'),
                          ('M Niece', 'mniece@email.com'),
                          ('A Child', 'achild@email.com'),
                          ('Mochi Dog', 'mdog@email.com')
                          ]
        connect_database.executemany(sql, customer_table)
    except ValueError as e:
        print('An error occurred: %s' % e)


# add_customer_data()


def add_or_update_customer_data():
    try:
        sql_update = "INSERT OR REPLACE INTO customer (name, email) VALUES ('John Doe', 'jdoe@email.com')"
        connect_database.execute(sql_update)
    except ValueError as e:
        print('An error occurred: %s' % e)


# add_or_update_customer_data()


# deletes any row with name and email = remove duplicate entries/customer data
def update_table_rows():
    try:
        delete_statement = 'DELETE FROM customer WHERE rowid > (SELECT MIN(rowid) FROM customer c2 WHERE customer.name = ' \
                           'c2.name AND customer.email = c2.email); '
        connect_database.execute(delete_statement)
    except ValueError as e:
        print('An error occurred: %s' % e)


# update_table_rows()


# print neatly  https://stackoverflow.com/questions/305378/list-of-tables-db-schema-dump-etc-using-the-python-sqlite3
# -api
def customer_data():
    try:
        select_all_table = "SELECT * FROM customer"
        cursor = connect_database.execute(select_all_table)
        results = cursor.fetchall()
        print(results)
    except ValueError as e:
        print('An error occurred: %s' % e)


print('customer database:')
customer_data()


###############################################################################
# order table initialized
###############################################################################
def create_order_table():
    try:
        connect_database.execute('''   
                      CREATE TABLE IF NOT EXISTS orders (
                     id INTEGER NOT NULL
            PRIMARY KEY AUTOINCREMENT,
                    status TEXT,
                  email TEXT,
                 notes TEXT,
                 product TEXT
                )
          ''')
    except ValueError as e:
        print('An error occurred: %s' % e)


create_order_table()


# added column to table order

def add_column_order():
    try:
        connect_database.execute("ALTER TABLE orders ADD product TEXT")
    except ValueError as e:
        print('An error occurred: %s' % e)


# add_column_order()


# use this to update OR add new customer specific ones can add later with UI
def add_order_data():
    try:
        sql = 'INSERT INTO orders( status, email, notes, product) values(?,?,?,?)'
        updated_order_table = [('en-route', 'jdoe@email.com', 'cc', 'earrings'),
                               ('pending', 'jhdoe@email.com', 'cc', 'ring'),
                               ('at hub', 'jdoe@email.com', 'wire', 'necklace'),
                               ('pending', 'jdoe@email.com', 'cc', 'bracelet'),
                               ('delayed', 'jdoe@email.com', 'cc', 'ring'),
                               ('delivered', 'kdog@email.com', 'cc', 'brooch'),
                               ('delivered', 'mniece@email.com', 'cc', 'necklace'),
                               ('delivered', 'achild@email.com', 'cc', 'ring'),
                               ('delivered', 'mdog@email.com', 'cc', 'ring')
                               ]

        connect_database.executemany(sql, updated_order_table)
    except ValueError as e:
        print('An error occurred: %s' % e)


# add_order_data()

# printing all orders
def order_data():
    try:
        select_all_table = "SELECT * FROM orders"
        cursor = connect_database.execute(select_all_table)
        results = cursor.fetchall()
        print(results)
    except ValueError as e:
        print('An error occurred: %s' % e)


print('orders database:')
order_data()


# https://www.krazyprogrammer.com/2020/12/how-to-search-data-from-sqlite-in.html

# remove duplicate entries
def update_table_rows_order():
    try:
        delete_statement = 'DELETE FROM orders WHERE rowid > (SELECT MIN(rowid) FROM orders o2 WHERE orders.email = ' \
                           'o2.email AND orders.id = o2.id); '
        connect_database.execute(delete_statement)
    except ValueError as e:
        print('An error occurred: %s' % e)


# update_table_rows_order()


# change line if need to update/insert
def add_or_update_order_data():
    try:
        sql_update = "INSERT OR REPLACE INTO orders (status, email, notes) VALUES ('at hub', 'jfoo@email.com', 'cc') "

        connect_database.execute(sql_update)
    except ValueError as e:
        print('An error occurred: %s' % e)


# add_or_update_order_data()

# remove duplicate orders
def delete_extra_entries():
    try:
        swl = "DELETE FROM orders WHERE email LIKE 'jfoo@email.com'"
        connect_database.execute(swl)
    except ValueError as e:
        print('An error occurred: %s' % e)


# delete_extra_entries()

# db commit
connect_database.commit()

#####################################################################################
# machine learning model
#####################################################################################


# Set file names for train and test data data import from CSV
test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')

# kaggle dataset customer complaints
csv_train_file = os.path.join(test_data_dir, 'complaints_processed.csv')

# created pseudo customer emails (the most common emails)
csv_test_file = os.path.join(test_data_dir, 'emails from Seattle Jewelry Company.csv')

# cleaned data for naiive bayes and doc2vec analysis
csv_tmp_file = os.path.join(test_data_dir, 'data_part_.csv')

# test emails i created that represent the majority of incoming customer communications
csv_test_result = os.path.join(test_data_dir, 'SJCompany.csv')

# test emails i created that represent the majority of outgoing customer communications
csv_response = os.path.join(test_data_dir, 'SJCCompanyOutgoingEmails.csv')
# saved model (saves 15 minutes of training time)
pickle_save = os.path.join(test_data_dir, 'eric_model.pkl')

# issue with values
# https://www.youtube.com/watch?v=OS2m0f2gVJ0
missing_narrative = ['N/a', "Nan", "NaN", np.nan, "na", "Na", None]

# needed to ignore first column (importing duplicate first col)
# https://www.statology.org/pandas-read-csv-ignore-first-column/

# read csv in chunks and put into a clean csv file for analysis
df_iterator = pd.read_csv(
    csv_train_file,
    chunksize=10000)

for i, df_chunk in enumerate(df_iterator):
    # Set writing mode to append after first chunk
    mode = 'w' if i == 0 else 'a'

    # Add header if it is the first chunk
    header = i == 0

    df_chunk.to_csv(
        csv_tmp_file,
        index=False,  # Skip index column
        header=header,
        mode=mode)

# dataframe df initialized
df = pd.read_csv(csv_tmp_file)

# expand column widths
pd.set_option('display.max_colwidth', None)

print(df)

# show value counts
print(df['product'].value_counts())

# remove null values
print(df.isnull().sum())
print(df.isnull().any())

# display NaN values and product number
nan_values = df[df['narrative'].isna()]

# show nan values
print(nan_values)

# drop wasn't working, needed to add parameter
# https://stackoverflow.com/questions/49712002/pandas-dropna-function-not-working
df.dropna(inplace=True)

# if email not in file-prompt customer to respond with email used to place order
# print out dataframe value counts making sure nan values were dropped
df['product'].value_counts().plot(kind='bar')

# bar plot labels and label orientation fixed (visual #1)
# plt.bar(x='product', height=3.0, width=3.0)
plt.xticks(rotation=10)
plt.title('Email Description Counts')
# plt.show()

# initialized dataframe complaints_dataframe for analysis by ML algorithms
complaints_dataframe = df[['product', 'narrative']]

# search for these terms and will use these for prediction and analysis later
search_terms = {'credit_reporting': 0, 'debt_collection': 1, 'mortgages_and_loans': 2, 'credit_card': 3,
                'retail_banking': 4
                }

# show value counts by product
print(complaints_dataframe['product'].value_counts())

# map search terms to products
complaints_dataframe['search_terms'] = complaints_dataframe['product'].map(search_terms)

# stemmer
stemmer = SnowballStemmer(language='english')

# stop words
stop_words = stopwords.words("english")

# load pickle model (saved model)
model = pickle.load(open(pickle_save, 'rb'))


def emails_out_to_customers():
    with open(csv_test_file, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if email_input in row:
                print(
                    "##################################################################################################")
                print("\nEmail exists in 'emails from Seattle Jewelry Company.csv' file")
                print("\nrows of data for given email:", row)
                # if email not in file-prompt customer to respond with email used to place order


# tokenizer - removes words less than 2 and ignores Xx
def tokenizer(text):
    token = [word for word in word_tokenize(text) if
             (len(word) > 3 and len(word.strip('Xx/')) > 2)]
    tokens = map(str.lower, token)
    stem = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
    return stem


# vectorizer
vectorize = TfidfVectorizer(analyzer=tokenizer)

# sets narrative to tfidf vectorizer
x_for = vectorize.fit_transform(df['narrative'][:1000].values.astype('U'))
print(complaints_dataframe.head(10))
print(complaints_dataframe.info(verbose=True))
# expand column size
pd.set_option('display.max_colwidth', None)

# use SMOTE for irregularly shaped data types
x_sm, y_sm = SMOTE().fit_resample(x_for, df['product'][:1000])

# initialize x and y train and test
X_train, X_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.3, random_state=0)

# checking shapes of each and theyre irregular- need SMOTE to fix
# issue with fit here
# https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/
# followed SMOTE guide because i ran into an error with size of data differences.

print('this is xtrain', X_train.shape)
print('this is xtest', X_test.shape)
print('this is y_train', y_train.shape)
print('this is y_test', y_test.shape)

# send data through multinomial naiive bayes algorithm
mnb = MultinomialNB()

# fit data to naiive bayes
mnb.fit(X_train, y_train)

# predict outcomes
X_test_predict = mnb.predict(X_test)
X_pred = mnb.predict(x_for)

# check classification
# print(classification_report(y_test, X_test_predict))

# check accuracy
print('MNB accuracy score: ', mnb.score(X_train, y_train))


###############################################################################
# read in text for doc2vec

def read_corpus(file, tokens_only=False):
    with smart_open.open(file, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


# train with kaggle dataset
train_corpus = list(read_corpus(csv_tmp_file))

# test with seattle jewelry company customer emails
test_corpus = list(read_corpus(csv_test_file))

# Train corpus print
print('this is train corpus', train_corpus[:2])

# Test corpus print
print('this is test corpus', test_corpus[:2])

###############################################################################
# Train model
# adding max_vocab_size=20000 to reduce memory issue
# https://stackoverflow.com/questions/59050644/memoryerror-unable-to-allocate-array-with-shape-and-data-type-float32-while-usi
# model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40, max_vocab_size=20000)

###############################################################################
# build vocab
# model.build_vocab(train_corpus)

###############################################################################
# check how often jewelry appears in train corpus
print(f"Word 'jewelry' appeared {model.wv.get_vecattr('jewelry', 'count')} times in train corpus.")

###############################################################################
# Train model
#
#
# model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

###############################################################################
# check trained model for terms as vectors

# list_of_terms = ['jewelry', 'pearls', 'necklace', 'earrings', 'gemstone']
# introductory- will expand terms in real life data with 'order status', 'exchange', 'return', 'refund'
# vector = model.infer_vector(list_of_terms)
# print(list_of_terms)
# print(vector)

# pickle save model
# pickle.dump(model, open(pickle_save, 'wb'))

# 2nd visual aide, Decision tree classifier
# dc = DecisionTreeClassifier()
# dc1 = dc.fit(X_train, y_train)
# y_predict = dc.predict(X_test)

# check accuracy of decision tree
# print("Decision Tree Classifier Accuracy check:", metrics.accuracy_score(y_test, y_predict))

# added tree plot and confusion matrix for display
# dc2 = tree.DecisionTreeClassifier(random_state=0)
# dcs2 = dc2.fit(X_train, y_train)
# tree.plot_tree(dcs2, fontsize=2)

# dcs = SVC(random_state=0)
# dcs.fit(X_train, y_train)
# SVC(random_state=0)
# 3rd visual aide confusion matrix display
# ConfusionMatrixDisplay.from_estimator(dcs, X_test, y_test)
# plt.show()

# Assessment of model ranking test vs train data

ranks = []
second_ranks = []
for doc_id in range(len(test_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

# document tank in train corpus

counter = collections.Counter(ranks)
print('this is ranking', counter)

doc_id = random.randint(0, len(train_corpus) - 1)

# Compare
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print('this is doc_id', doc_id)
sim_id = random.randint(0, len(train_corpus) - 1)
print('this is sim_id', sim_id)
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id].words)))

# Pick a random doc from test to compare
doc_id2 = random.randint(0, len(test_corpus) - 1)
print('this is docid2', doc_id2)
inferred_vect = model.infer_vector(test_corpus[doc_id2].words)
sims = model.dv.most_similar([inferred_vect], topn=len(model.dv))

# added .words after 426 and 431.
# Compare to train corpus
print('Test Document ({}): «{}»\n'.format(doc_id2, ' '.join(test_corpus[doc_id2].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

# check to see if model agrees
print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


# with open (csv_test_file, 'rt) as f
# mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
# if st.find(b'jhdoe@email.com') != -1:


#    print('true')
#    data_frame = ['jhdoe@email.com']
#   vector = vectorize.fit_transform(data_frame)
#   update = vector.toarray()
#  this = pd.DataFrame(update, columns=y_test)
#  category = mnb.predict(this)

#  print("this is category value:", category)
#  print(
#    "##################################################################################################")
#   print("\nEmail exists in 'emails from Seattle Jewelry Company.csv' file")
# print("\nrows of data for given email:", row)

# def customer_product_check():
def find_index(input):
    o = open(csv_test_file, 'r')
    my_data = csv.reader(o)
    index = 0
    for row in my_data:
        # print row
        if row[0] == input:
            return index
        else:
            index += 1


enc = OneHotEncoder(handle_unknown='ignore')
# creating instance of one-hot-encoder

# passing ['search_terms'] column (label encoded values of ['search_terms']s)
enc_df = pd.DataFrame(enc.fit_transform(complaints_dataframe[['search_terms']]).toarray())
# merge with main df complaints_dataframe on key values
complaints_dataframe = complaints_dataframe.join(enc_df)


def check_for_email(email_input):
    with open(csv_test_file, 'r+') as af:

        line = af.readlines()
        for row in line:

            if email_input in row:
                print("\nEmail exists in 'emails from Seattle Jewelry Company.csv' file")
                print("\nrows of data for given email:", row)

                # newshape = category.reshape(1, -1)
                # merge with main df bridge_df on key values
                df_email_input = [row[:]]
                # category = mnb.predict(np.array([email_input]).reshape(-1, 1))
                # print(category)
                predict_vector = vectorize.fit_transform(df_email_input)
                p_to_array = predict_vector.toarray()
                cat = pd.DataFrame(enc.fit_transform(p_to_array))
                p_to_array = p_to_array.join(cat)
                #cat_up = category.to_xarray()
                #cat = cat_up.to_array()

                predict_this = mnb.predict(p_to_array)
                print(predict_this)
                # if email not in file-prompt +?ustomer to respond with email used to place order


###############################################################################
############################# GUI #############################################
###############################################################################
print(
    "########################################################################################################################")

if __name__ == '__main__':
    print("Computer Science Capstone C964 | Nicole Mau | nmau@wgu.edu | "
          "001336361 | eric_bot | email_response_in_corporations bot")

    # loop until user is satisfied
    isExit = True
    while isExit:
        print("\nOptions:")
        print("1. Print All Order Data")
        print("2. Get a Specific Order Status with ID")
        print("3. Get all Customer Data")
        print("4. Get Specific Customer/order Information by email")
        print("5. Add New Customer")
        print("6. Exit the Program")
        option = input("Chose an option (1,2,3,4,5 or 6): ")
        # print all order data
        if option == "1":
            order_data()

            # print order info from specific order ID
        elif option == "2":
            print("Please enter your order ID")
            orderID = input(" ")
            cursor = connect_database.execute("SELECT * FROM ORDERS where id = ?", (orderID,))
            results = cursor.fetchall()
            print(results)

        # print all customer data
        elif option == "3":

            customer_data()
        # print specific order data by email
        elif option == "4":
            print("Please enter customer email")
            email_input = input(" ")
            c2 = connect_database.execute("SELECT * FROM CUSTOMER where email = ? ", (email_input,))
            result = c2.fetchall()
            print("Customer information for email address:", result)
            cursor = connect_database.execute("SELECT * FROM ORDERS where email = ? ", (email_input,))
            results = cursor.fetchall()
            print("Order information for selected email:", results)

            check_for_email(email_input)

            # https://stackoverflow.com/questions/17308872/check-whether-string-is-in-csv

            # doc_id3 = find_index(email_input)
            # print(doc_id3)
            # inferred_vectors = model.infer_vector(test_corpus[doc_id3].words)
            # sim2 = model.dv.most_similar([inferred_vect], topn=len(model.dv))
            # if email not in file-prompt customer to respond with email used to place order
        # option to exit
        elif option == '5':
            print("Please enter customer name")
            name_input = input("")
            print("Please enter customer email")
            email_input = input(" ")

            c3 = connect_database.execute("INSERT INTO customer (name, email) values (?,?)", (name_input, email_input))
            result = c3.fetchall()
            print("New Customer Added to database:", result)
        elif option == '6':
            isExit = False
        else:
            print("Invalid option, please try again!")
        # main - END
