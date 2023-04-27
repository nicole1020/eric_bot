r"""
Doc2Vec Model
=============

Introduces Gensim's Doc2Vec model and demonstrates its use on the
`Lee Corpus <https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`__.

"""
import csv
import logging
import pickle

import smart_open
from imblearn.over_sampling import SMOTE
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# Doc2Vec is a :ref:`core_concepts_model` that represents each
# :ref:`core_concepts_document` as a :ref:`core_concepts_vector`.  This
# tutorial introduces the model and demonstrates how to train and assess it.
#
# Here's a list of what we'll be doing:
#
# 0. Review the relevant models: bag-of-words, Word2Vec, Doc2Vec
# 1. Load and preprocess the training and test corpora (see :ref:`core_concepts_corpus`)
# 2. Train a Doc2Vec :ref:`core_concepts_model` model using the training corpus
# 3. Demonstrate how the trained model can be used to infer a :ref:`core_concepts_vector`
# 4. Assess the model
# 5. Test the model on the test corpus
#
# Review: Bag-of-words
# --------------------
#
# .. Note:: Feel free to skip these review sections if you're already familiar with the models.
#
# You may be familiar with the `bag-of-words model
# <https://en.wikipedia.org/wiki/Bag-of-words_model>`_ from the
# :ref:`core_concepts_vector` section.
# This model transforms each document to a fixed-length vector of integers.
# For example, given the sentences:
#
# - ``John likes to watch movies. Mary likes movies too.``
# - ``John also likes to watch football games. Mary hates football.``
#
# The model outputs the vectors:
#
# - ``[1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0]``
# - ``[1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1]``
#
# Each vector has 10 elements, where each element counts the number of times a
# particular word occurred in the document.
# The order of elements is arbitrary.
# In the example above, the order of the elements corresponds to the words:
# ``["John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games", "hates"]``.
#
# Bag-of-words models are surprisingly effective, but have several weaknesses.
#
# First, they lose all information about word order: "John likes Mary" and
# "Mary likes John" correspond to identical vectors. There is a solution: bag
# of `n-grams <https://en.wikipedia.org/wiki/N-gram>`__
# models consider word phrases of length n to represent documents as
# fixed-length vectors to capture local word order but suffer from data
# sparsity and high dimensionality.
#
# Second, the model does not attempt to learn the meaning of the underlying
# words, and as a consequence, the distance between vectors doesn't always
# reflect the difference in meaning.  The ``Word2Vec`` model addresses this
# second problem.
#
# Review: ``Word2Vec`` Model
# --------------------------
#
# ``Word2Vec`` is a more recent model that embeds words in a lower-dimensional
# vector space using a shallow neural network. The result is a set of
# word-vectors where vectors close together in vector space have similar
# meanings based on context, and word-vectors distant to each other have
# differing meanings. For example, ``strong`` and ``powerful`` would be close
# together and ``strong`` and ``Paris`` would be relatively far.
#
# Gensim's :py:class:`~gensim.models.word2vec.Word2Vec` class implements this model.
#
# With the ``Word2Vec`` model, we can calculate the vectors for each **word** in a document.
# But what if we want to calculate a vector for the **entire document**\ ?
# We could average the vectors for each word in the document - while this is quick and crude, it can often be useful.
# However, there is a better way...
#
# Introducing: Paragraph Vector
# -----------------------------
#
# .. Important:: In Gensim, we refer to the Paragraph Vector model as ``Doc2Vec``.
#
# Le and Mikolov in 2014 introduced the `Doc2Vec algorithm <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`__,
# which usually outperforms such simple-averaging of ``Word2Vec`` vectors.
#
# The basic idea is: act as if a document has another floating word-like
# vector, which contributes to all training predictions, and is updated like
# other word-vectors, but we will call it a doc-vector. Gensim's
# :py:class:`~gensim.models.doc2vec.Doc2Vec` class implements this algorithm.
#
# There are two implementations:
#
# 1. Paragraph Vector - Distributed Memory (PV-DM)
# 2. Paragraph Vector - Distributed Bag of Words (PV-DBOW)
#
# .. Important::
#   Don't let the implementation details below scare you.
#   They're advanced material: if it's too much, then move on to the next section.
#
# PV-DM is analogous to Word2Vec CBOW. The doc-vectors are obtained by training
# a neural network on the synthetic task of predicting a center word based an
# average of both context word-vectors and the full document's doc-vector.
#
# PV-DBOW is analogous to Word2Vec SG. The doc-vectors are obtained by training
# a neural network on the synthetic task of predicting a target word just from
# the full document's doc-vector. (It is also common to combine this with
# skip-gram testing, using both the doc-vector and nearby word-vectors to
# predict a single target word, but only one at a time.)
#
# Prepare the Training and Test Data
# ----------------------------------
#
# For this tutorial, we'll be training our model using the `Lee Background
# Corpus
# <https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`_
# included in gensim. This corpus contains 314 documents selected from the
# Australian Broadcasting Corporation’s news mail service, which provides text
# e-mails of headline stories and covers a number of broad topics.
#
# And we'll test our model by eye using the much shorter `Lee Corpus
# <https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`_
# which contains 50 documents.
#

import os
import gensim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB

# Set file names for train and test data
test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
csv_train_file = os.path.join(test_data_dir, 'complaints_processed.csv')
csv_test_file = os.path.join(test_data_dir, 'emails from Seattle Jewelry Company.csv')
csv_tmp_file = os.path.join(test_data_dir, 'data_part_.csv')
csv_test_result = os.path.join(test_data_dir, 'SJCompany.csv')

pickle_save = os.path.join(test_data_dir, 'eric_model.sav')
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


def tokenizer(text):
    token = [word for word in word_tokenize(text) if
             (len(word) > 3 and len(word.strip('Xx/')) > 2)]
    tokens = map(str.lower, token)
    stem = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
    return stem


vectorize = TfidfVectorizer(analyzer=tokenizer)
x_for = vectorize.fit_transform(df['narrative'][:8219].values.astype('U'))

pd.set_option('display.max_colwidth', None)
x_sm, y_sm = SMOTE().fit_resample(x_for, df['product'][:8219])
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

from sklearn.metrics import classification_report

print(classification_report(y_test, X_test_predict))


###############################################################################
# Define a Function to Read and Preprocess Text
# ---------------------------------------------
#
# Below, we define a function to:
#
# - open the train/test file (with latin encoding)
# - read the file line-by-line
# - pre-process each line (tokenize text into individual words, remove punctuation, set to lowercase, etc)
#
# The file we're reading is a **corpus**.
# Each line of the file is a **document**.
#
# .. Important::
#   To train the model, we'll need to associate a tag/number with each document
#   of the training corpus. In our case, the tag is simply the zero-based line
#   number.


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

###############################################################################
# Let's take a look at the training corpus
#
print('this is train corpus', train_corpus[:2])

###############################################################################
# And the testing corpus looks like this:
#
print('this is test corpus', test_corpus[:2])

###############################################################################
# Notice that the testing corpus is just a list of lists and does not contain
# any tags.
#

###############################################################################
# Training the Model
# ------------------
#
# Now, we'll instantiate a Doc2Vec model with a vector size with 50 dimensions and
# iterating over the training corpus 40 times. We set the minimum word count to
# 2 in order to discard words with very few occurrences. (Without a variety of
# representative examples, retaining such infrequent words can often make a
# model worse!) Typical iteration counts in the published `Paragraph Vector paper <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`__
# results, using 10s-of-thousands to millions of docs, are 10-20. More
# iterations take more time and eventually reach a point of diminishing
# returns.
#
# However, this is a very very small dataset (300 documents) with shortish
# documents (a few hundred words). Adding training passes can sometimes help
# with such small datasets.
# adding max_vocab_size=50000 to reduce memory issue and size=300
# https://stackoverflow.com/questions/59050644/memoryerror-unable-to-allocate-array-with-shape-and-data-type-float32-while-usi
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40, max_vocab_size=20000)

###############################################################################
# Build a vocabulary
model.build_vocab(train_corpus)

###############################################################################
# Essentially, the vocabulary is a list (accessible via
# ``model.wv.index_to_key``) of all of the unique words extracted from the training corpus.
# Additional attributes for each word are available using the ``model.wv.get_vecattr()`` method,
# For example, to see how many times ``penalty`` appeared in the training corpus:
#
print(f"Word 'jewelry' appeared {model.wv.get_vecattr('jewelry', 'count')} times in train corpus.")

###############################################################################
# Next, train the model on the corpus.
# In the usual case, where Gensim installation found a BLAS library for optimized
# bulk vector operations, this training on this tiny 300 document, ~60k word corpus
# should take just a few seconds. (More realistic datasets of tens-of-millions
# of words or more take proportionately longer.) If for some reason a BLAS library
# isn't available, training uses a fallback approach that takes 60x-120x longer,
# so even this tiny training will take minutes rather than seconds. (And, in that
# case, you should also notice a warning in the logging letting you know there's
# something worth fixing.) So, be sure your installation uses the BLAS-optimized
# Gensim if you value your time.
#
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

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

pickle.dump(model, open(pickle_save, 'wb'))

dc = DecisionTreeClassifier()
dc = dc.fit(X_train, y_train)
y_predict = dc.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
plt.show()

###############################################################################
# Note that ``infer_vector()`` does *not* take a string, but rather a list of
# string tokens, which should have already been tokenized the same way as the
# ``words`` property of original training document objects.
#
# Also note that because the underlying training/inference algorithms are an
# iterative approximation problem that makes use of internal randomization,
# repeated inferences of the same text will return slightly different vectors.
#

###############################################################################
# Assessing the Model
# -------------------
#
# To assess our new model, we'll first infer new vectors for each document of
# the training corpus, compare the inferred vectors with the training corpus,
# and then returning the rank of the document based on self-similarity.
# Basically, we're pretending as if the training corpus is some new unseen data
# and then seeing how they compare with the trained model. The expectation is
# that we've likely overfit our model (i.e., all of the ranks will be less than
# 2) and so we should be able to find similar documents very easily.
# Additionally, we'll keep track of the second ranks for a comparison of less
# similar documents.
#
ranks = []
second_ranks = []
for doc_id in range(len(test_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

###############################################################################
# Let's count how each document ranks with respect to the training corpus
#
# NB. Results vary between runs due to random seeding and very small corpus
import collections

counter = collections.Counter(ranks)
print('this is ranking', counter)

import random

doc_id = random.randint(0, len(train_corpus) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print('this is doc_id', doc_id)
sim_id = random.randint(0, len(train_corpus) - 1)
print('this is sim_id', sim_id)
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id].words)))

# Pick a random document from the test corpus and infer a vector from the model testing model
doc_id2 = random.randint(0, len(test_corpus) - 1)
print('this is docid2', doc_id2)
inferred_vect = model.infer_vector(test_corpus[doc_id2].words)
sims = model.dv.most_similar([inferred_vect], topn=len(model.dv))

# added .words after 426 and 431.
# Compare and print the most/median/the least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id2].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
###############################################################################
# Conclusion
# ----------
#
# Let's review what we've seen in this tutorial:
#
# 0. Review the relevant models: bag-of-words, Word2Vec, Doc2Vec
# 1. Load and preprocess the training and test corpora (see :ref:`core_concepts_corpus`)
# 2. Train a Doc2Vec :ref:`core_concepts_model` model using the training corpus
# 3. Demonstrate how the trained model can be used to infer a :ref:`core_concepts_vector`
# 4. Assess the model
# 5. Test the model on the test corpus
#
# That's it! Doc2Vec is a great way to explore relationships between documents.
#
# Additional Resources
# --------------------
#
# If you'd like to know more about the subject matter of this tutorial, check out the links below.
#
# * `Word2Vec Paper <https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf>`_
# * `Doc2Vec Paper <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`_
# * `Dr. Michael D. Lee's Website <http://faculty.sites.uci.edu/mdlee>`_
# * `Lee Corpus <http://faculty.sites.uci.edu/mdlee/similarity-data/>`__
# * `IMDB Doc2Vec Tutorial <doc2vec-IMDB.ipynb>`_
#
