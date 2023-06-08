import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from collections import Counter
import nltk
import itertools
import re
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import nltk

class LDA:
    
    def __init__(self, data):
        self.ALPHA = 0.1
        self.BETA = 0.1
        self.NUM_TOPICS = 5
        self.data = data

    def generate_frequencies(self, data, max_docs=None):
        freqs = Counter()
        all_stopwords = set(stopwords.words('english'))
        all_stopwords.add("enron")
        nr_tokens = 0
        tokens = [token.lower() for doc in data[:max_docs] 
                for token in itertools.chain(*[word_tokenize(doc)]) 
                if token.lower() not in all_stopwords and token.isalpha()]
        
        freqs.update(tokens)
        return freqs

    def get_vocab(self, freqs, freq_threshold=8):
        vocab = {word: idx for idx, word in enumerate(word for word in freqs if freqs[word] >= freq_threshold)}
        vocab_idx_str = {idx: word for word, idx in vocab.items()}
        return vocab, vocab_idx_str

    def tokenize_dataset(self, data, vocab, max_docs=10000):
        docs = [ [vocab[token.lower()] for token in word_tokenize(doc) if token.lower() in vocab] 
                for doc in data[:max_docs] ]
        corpus = [np.asarray(doc) for doc in docs]
        return docs, corpus
    
    def LDA_Collapsed_Gibbs(self, corpus, num_iter=200):
        Z = []

        num_docs = len(corpus)

        for _, doc in enumerate(corpus):
            Zd = np.random.randint(low=0, high=self.NUM_TOPICS, size=(len(doc)))
            Z.append(Zd)

        ndk = np.zeros((num_docs, self.NUM_TOPICS))
        for d in range(num_docs):
            for k in range(self.NUM_TOPICS):
                ndk[d, k] = np.sum(Z[d] == k)

        nkw = np.zeros((self.NUM_TOPICS, self.vocab_size))
        for doc_idx, doc in enumerate(corpus):
            for i, word in enumerate(doc):
                topic = Z[doc_idx][i]
                nkw[topic, word] += 1

        nk = np.sum(nkw, axis=1)
        topic_list = [i for i in range(self.NUM_TOPICS)]
        prob_list = []

        ap = False
        for iter in range(num_iter):
            if iter == (num_iter - 1):
                ap = True
            if len(prob_list) > 0:
                ap = False
            for doc_idx, doc in enumerate(corpus):
                doc_prof = np.zeros((1, self.NUM_TOPICS))
                for i in range(len(doc)):
                    word = doc[i]
                    topic = Z[doc_idx][i]
                    ndk[doc_idx, topic] -= 1
                    nkw[topic, word] -= 1
                    nk[topic] -= 1
                    p_z = (ndk[doc_idx, :] + self.ALPHA) * (nkw[:, word] + self.BETA) / (nk[:] + self.BETA * self.vocab_size)
                    topic = random.choices(topic_list, weights=p_z, k=1)[0]
                    if ap:
                        doc_prof += p_z
                    Z[doc_idx][i] = topic
                    ndk[doc_idx, topic] += 1
                    nkw[topic, word] += 1
                    nk[topic] += 1
                if ap:
                    doc_prof = doc_prof / len(doc)
                    prob_list.append(doc_prof)

        return Z, ndk, nkw, nk, prob_list
    
    def produce_keywords(self):
        freqs = self.generate_frequencies(self.data)
        vocab, vocab_idx_str = self.get_vocab(freqs)
        docs, corpus = self.tokenize_dataset(self.data, vocab)
        self.vocab_size = len(vocab)
        Z, ndk, nkw, nk, prob_list1 = self.LDA_Collapsed_Gibbs(corpus)

        phi = nkw / nk.reshape(self.NUM_TOPICS, 1)

        num_words = 5
        keywords = []
        for k in range(self.NUM_TOPICS):
            top_words_indices = np.argsort(phi[k])[::-1][:num_words]
            top_words = [vocab_idx_str[word_index] for word_index in top_words_indices]
            filtered_words = [word for word in top_words if len(word) > 3 and nltk.pos_tag([word])[0][1] == 'NN']
            keywords.extend(filtered_words)

        return keywords
    
    
    def only_nouns(text):
        li = nltk.pos_tag(text)
        new_li = []
        for i in range(len(text)):
            if li[i][1] == 'NN' or '_' in li[i][0]:
                new_li.append(text[i])
        return new_li

def calculate_min_freq(text_size):
    if text_size < 1000:
        min_freq = 5
    elif text_size < 5000:
        min_freq = 10
    elif text_size < 10000:
        min_freq = 15
    else:
        min_freq = 20
    
    return min_freq

def find_ngrams_keywords(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub("[^A-Za-z ]","",text)
    words = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    finder = BigramCollocationFinder.from_words(filtered_words)

    finder.apply_freq_filter(5)

    bigram_measures = BigramAssocMeasures()
    top_bigrams = finder.nbest(bigram_measures.chi_sq, 30)

    bigrams = []
    for bigram in top_bigrams:
        bigrams.append(bigram[0] + ' ' + bigram[1])
    if len(bigrams) > 0:
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        doc_embedding = model.encode([text])
        candidate_embeddings1 = model.encode(bigrams)

        top_n = 10
        distances = cosine_similarity(doc_embedding, candidate_embeddings1)
        keywords = [bigrams[index] for index in distances.argsort()[0][-top_n:]]
    else:
        keywords = []
    return keywords

def key_words_extraction(text):
    text_lda = text.split(' ')
    df = pd.DataFrame()
    df['text'] = text_lda
    lda = LDA(df['text'].values)
    lda_keywords = lda.produce_keywords()
    ngrams_keywords = find_ngrams_keywords(text)
    print(ngrams_keywords)
    all_keywords = ", ".join(list(set(lda_keywords + ngrams_keywords)))
    return all_keywords

