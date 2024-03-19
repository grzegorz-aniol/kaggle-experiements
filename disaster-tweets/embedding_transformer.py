import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize

class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=50, epochs=20):
        self.vector_size = vector_size
        self.epochs = epochs
        self.model = None

    def fit(self, X, y=None):
        tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(X)]
        self.model = Doc2Vec(vector_size=self.vector_size, min_count=1, epochs=self.epochs)
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self

    def transform(self, X, y=None):
        return np.array([self.model.infer_vector(word_tokenize(doc.lower())) for doc in X])
