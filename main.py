"""
"Computer Science Capstone C964 | Nicole Mau | nmau@wgu.edu | "
          "001336361 | eric_bot | email response in corporations"
"""
import logging
import pickle
from datetime import timedelta

import smart_open
from imblearn.over_sampling import SMOTE
from sklearn import metrics, tree
from sklearn.metrics import ConfusionMatrixDisplay
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
import sqlite3 as sl
from sklearn.metrics import classification_report
import collections
import random
from tkinter import *
import tkinter.ttk as ttk

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# create sqllite sql database for future sensitive customer data

connect_database = sl.connect('my_test_customers.db')


###############################################################################
# create table in db running into issues with .execute command
# https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-execute.html


# connect to local database create table in db running into issues with .execute command
# https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-execute.html
# https://stackoverflow.com/questions/53128279/how-to-print-output-from-sqlite3-in-python
#

# reset table to make it clean again
# def clear_customer_data():
#  delete_all = "DELETE FROM ORDERS"
# connect_database.execute(delete_all)


# clear_customer_data()
# def add_customer_data():
#   sql = 'INSERT INTO CUSTOMER( name, email, notes) values(?,?,?)'
#    customer_table = [('John Doe', 'jdoe@email.com', 'order number 11'),
#             ('Johnny Doe', 'jhdoe@email.com', 'order number 12'),
#              ('Jane Doe', 'jdoe@email.com', 'order number 13'),
#               ('Janey Do', 'jdoe@email.com', 'order number 14'),
#                ('Joey Doe', 'jdoe@email.com', 'order number 15'),
#                 ('Katsu Dog', 'kdog@email.com', 'order number 16'),
#                  ('M Niece', 'mniece@email.com', 'order number 17'),
#                   ('A Child', 'achild@email.com', 'order number 18'),
#                    ('Mochi Dog', 'mdog@email.com', 'order number 19')
#                     ]

#    connect_database.executemany(sql, customer_table)


# add_customer_data()


# def add_order_data():
# sql = 'INSERT INTO ORDERS( status, email, notes) values(?,?,?)'
# order_table = [('en-route', 'jdoe@email.com', 'cc'),
#         ('pending', 'jhdoe@email.com', 'cc'),
#          ('at hub', 'jdoe@email.com', 'wire'),
#           ('pending', 'jdoe@email.com', 'cc'),
#             ('delayed', 'jdoe@email.com', 'cc'),
#             ('delivered', 'kdog@email.com', 'cc'),
#              ('delivered', 'mniece@email.com', 'cc'),
#               ('delivered', 'achild@email.com', 'cc'),
#                ('delivered', 'mdog@email.com', 'cc')
#                 ]

#   connect_database.executemany(sql, order_table)


# add_order_data()
###############################################################################

def create_customer_table():
    connect_database.execute('''
                  CREATE TABLE IF NOT EXISTS CUSTOMER (
                 id INTEGER NOT NULL
        PRIMARY KEY AUTOINCREMENT,
               name TEXT,
              email TEXT,
             notes TEXT
            )
      ''')


create_customer_table()


# use this to update OR add new customer specific ones can add later with UI

def add_or_update_customer_data():
    sql_update = "INSERT OR REPLACE INTO CUSTOMER (name, email, notes) VALUES ('John Foo', 'jfoo@email.com', " \
                 "'order number 20') "

    connect_database.execute(sql_update)


add_or_update_customer_data()


# deletes any row with name and email = remove duplicate entries/customer data
def update_table_rows():
    delete_statement = 'DELETE FROM CUSTOMER WHERE rowid > (SELECT MIN(rowid) FROM CUSTOMER c2 WHERE CUSTOMER.name = ' \
                       'c2.name AND CUSTOMER.email = c2.email); '
    connect_database.execute(delete_statement)


update_table_rows()


# print neatly  https://stackoverflow.com/questions/305378/list-of-tables-db-schema-dump-etc-using-the-python-sqlite3
# -api
def customer_data():
    select_all_table = "SELECT * FROM CUSTOMER"
    cursor = connect_database.execute(select_all_table)
    results = cursor.fetchall()
    print(results)


print('customer database:')
customer_data()


def create_order_table():
    connect_database.execute('''   
                      CREATE TABLE IF NOT EXISTS ORDERS (
                     id INTEGER NOT NULL
            PRIMARY KEY AUTOINCREMENT,
                    status TEXT,
                  email TEXT,
                 notes TEXT
                )
          ''')


create_order_table()


def order_data():
    select_all_table = "SELECT * FROM ORDERS"
    cursor = connect_database.execute(select_all_table)
    results = cursor.fetchall()
    print(results)


print('orders database:')
order_data()


# https://www.krazyprogrammer.com/2020/12/how-to-search-data-from-sqlite-in.html


def update_table_rows_order():
    delete_statement = 'DELETE FROM ORDERS WHERE rowid > (SELECT MIN(rowid) FROM ORDERS o2 WHERE ORDERS.email = ' \
                       'o2.email AND ORDERS.id = o2.id); '
    connect_database.execute(delete_statement)


update_table_rows_order()


def add_or_update_order_data():
    sql_update = "INSERT OR REPLACE INTO ORDERS (status, email, notes) VALUES ('at hub', 'jfoo@email.com', 'cc') "

    connect_database.execute(sql_update)


add_or_update_order_data()

connect_database.commit()
#####################################################################################
# Set file names for train and test data
test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
csv_train_file = os.path.join(test_data_dir, 'complaints_processed.csv')
csv_test_file = os.path.join(test_data_dir, 'emails from Seattle Jewelry Company.csv')
csv_tmp_file = os.path.join(test_data_dir, 'data_part_.csv')
csv_test_result = os.path.join(test_data_dir, 'SJCompany.csv')

pickle_save = os.path.join(test_data_dir, 'eric_model.pkl')
# issue with values
# https://www.youtube.com/watch?v=OS2m0f2gVJ0
missing_narrative = ['N/a', "Nan", "NaN", np.nan, "na", "Na", None]

# needed to ignore first column (importing duplicate first col)
# https://www.statology.org/pandas-read-csv-ignore-first-column/

# print('this is csv_tmp_file', csv_tmp_file)

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

df = pd.read_csv(csv_tmp_file)
pd.set_option('display.max_colwidth', None)
plt.show()
print(df)
print(df['product'].value_counts())
print(df.isnull().sum())
print(df.isnull().any())
# display NaN values and product number
nan_values = df[df['narrative'].isna()]
print(nan_values)
# drop wasn't working, needed to add parameter
# https://stackoverflow.com/questions/49712002/pandas-dropna-function-not-working
df.dropna(inplace=True)

df['product'].value_counts().plot(kind='bar')
plt.bar(x='product', height=3.0, width=3.0)
plt.xticks(rotation=10)
plt.title('Email Description Counts')
plt.show()

# check if drop worked
complaints_dataframe = df[['product', 'narrative']]
search_terms = {'credit_reporting': 0, 'debt_collection': 1, 'mortgages_and_loans': 2, 'credit_card': 3,
                'retail_banking': 4
                }

print(complaints_dataframe['product'].value_counts())

complaints_dataframe['search_terms'] = complaints_dataframe['product'].map(search_terms)
stemmer = SnowballStemmer(language='english')

stop_words = stopwords.words("english")

# load pickle model
model = pickle.load(open(pickle_save, 'rb'))


def tokenizer(text):
    token = [word for word in word_tokenize(text) if
             (len(word) > 3 and len(word.strip('Xx/')) > 2)]
    tokens = map(str.lower, token)
    stem = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
    return stem


vectorize = TfidfVectorizer(analyzer=tokenizer)
x_for = vectorize.fit_transform(df['narrative'][:10000].values.astype('U'))

pd.set_option('display.max_colwidth', None)
x_sm, y_sm = SMOTE().fit_resample(x_for, df['product'][:10000])
# X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.3, random_state=0)
print('this is xtrain', X_train.shape)
print('this is xtest', X_test.shape)
print('this is y_train', y_train.shape)
print('this is y_test', y_test.shape)

# issue with fit here
# https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/
# followed SMOTE guide because i ran into an error with size of data differences.

mnb = MultinomialNB()

mnb.fit(X_train, y_train)

X_test_predict = mnb.predict(X_test)
X_pred = mnb.predict(x_for)

print(classification_report(y_test, X_test_predict))


###############################################################################
# read in text

def read_corpus(file, tokens_only=False):
    with smart_open.open(file, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


train_corpus = list(read_corpus(csv_tmp_file))
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
# Now, we can use the trained model to infer a vector for any piece of text
# by passing a list of words to the ``model.infer_vector`` function. This
# vector can then be compared with other vectors via cosine similarity.
#
list_of_terms = ['jewelry', 'pearls', 'necklace', 'earrings', 'gemstone']
# introductory- will expand terms in real life data with 'order status', 'exchange', 'return', 'refund'
vector = model.infer_vector(list_of_terms)
print(list_of_terms)
print(vector)

# pickle save model
# pickle.dump(model, open(pickle_save, 'wb'))

dc = DecisionTreeClassifier()
dc1 = dc.fit(X_train, y_train)
y_predict = dc.predict(X_test)
print("Accuracy check:", metrics.accuracy_score(y_test, y_predict))

dc2 = tree.DecisionTreeClassifier(random_state=0)
dcs2 = dc2.fit(X_train, y_train)
tree.plot_tree(dcs2)
# added tree plot and confusion matrix for display

dcs = SVC(random_state=0)
dcs.fit(X_train, y_train)
SVC(random_state=0)
ConfusionMatrixDisplay.from_estimator(dcs, X_test, y_test)
plt.show()

# Assessment of model

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
for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
###############################################################################
###############################################################################


if __name__ == '__main__':
    print("Computer Science Capstone C964 | Nicole Mau | nmau@wgu.edu | "
          "001336361 | eric_bot | email response in corporations")

    # loop until user is satisfied
    isExit = True
    while isExit:
        print("\nOptions:")
        print("1. Print All Orders in Table")
        print("2. Get a Single Order Status with ID")
        print("3. Get a single Order Status with a Time")
        print("4. Exit the Program")
        option = input("Chose an option (1,2,3, or 4): ")
        if option == "1":
            order_data()

            # print order info from ID
        elif option == "2":
            print("Please enter your order ID")
            orderID = input(" ")
            cur = connect_database.cursor()
            cur.execute("SELECT * FROM ORDERS WHERE id=?", (orderID,))
            rows = cur.fetchall()


        elif option == "3":
            print("Please enter your customer id")
            customer_id = input(" ")
            print("Please enter your email")
            email_input = input(" ")
            cur = connect_database.cursor()
            cur.execute("SELECT status FROM (  SELECT * FROM ORDERS UNION ALL SELECT email, name FROM CUSTOMERS) "
                        "WHERE email = :email", (customer_id, email_input))
            rows = cur.fetchall()


        elif option == '4':
            isExit = False
        else:
            print("Invalid option, please try again!")
        # main - END
