import pickle

import gensim
import smart_open

from main import pickle_save, csv_tmp_file, csv_test_file, X_test, y_test

# utilize crisp dm
# https://towardsdatascience.com/crisp-dm-methodology-for-your-first-data-science-project-769f35e0346c

loaded_model = pickle.load(open(pickle_save, 'rb'))
result = loaded_model.score(X_test, y_test)
print('result from pickle', result)


def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


train_corpus = list(read_corpus(csv_tmp_file))
test_corpus = list(read_corpus(csv_test_file, tokens_only=True))

###############################################################################
# Let's take a look at the training corpus
#
print(train_corpus[:2])

###############################################################################
# And the testing corpus looks like this:
#
print(test_corpus[:2])

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
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40, max_vocab_size=50000)

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
for doc_id in range(len(train_corpus)):
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
print(counter)

import random

doc_id = random.randint(0, len(train_corpus) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))

# Pick a random document from the test corpus and infer a vector from the model testing model
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

# Compare and print the most/median/the least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
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
# load the model from disk
