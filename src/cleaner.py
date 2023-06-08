import re
import nltk
from collections import Counter
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class TextCleaner:

    vocab = set()
    words_frequences = {}
    remove_words = set()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    data = ''
    
    def __init__(self, corpus):
        self.corpus = corpus

    def basic_cleaning(self, text):
        text = re.sub("[^A-Za-z ]","",text)
        text = re.sub(' +', ' ', text)
        text = text.lower()
        text = text.split()
        text = [w for w in text if w not in self.stop_words]
        text = [self.lemmatizer.lemmatize(word) for word in text]
        #text = " ".join(text)
        return text
    
    def apply_basic_cleaning(self):
        self.corpus = self.corpus.apply(lambda x: self.basic_cleaning(x))
        
    def replace_ngram(self, x, ngram_array):
        for gram in ngram_array:
            x = x.replace(gram, '_'.join(gram.split()))
        return x

    def form_vocab(self):
        self.vocab.clear()
        for i in self.corpus:
            for j in i:
                self.vocab.add(j)
    
    def build_data(self):
        d = {i : [] for i in list(self.vocab)}
        self.data = pd.DataFrame(d)

        for doc in self.corpus:
            self.data = self.data.append({i : doc.count(i) for i in list(self.vocab)}, ignore_index=True)
        self.data = self.data.transpose()

    def calc_frequences(self):
        self.words_frequences = dict.fromkeys(list(self.vocab), 0)
        for word in list(self.vocab):
            for i in self.data.loc[[word], :].sum():
                self.words_frequences[word] += i
        self.words_frequences = {k: v for k, v in sorted(self.words_frequences.items(), key=lambda item: item[1], reverse=True)}

    def find_bigrams_trigrams(self):
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder =nltk.collocations.BigramCollocationFinder.from_documents(self.corpus)
        finder.apply_freq_filter(30)
        bigram_scores = finder.score_ngrams(bigram_measures.pmi)
        bigrams = [" ".join(list(bigram[0])) for bigram in bigram_scores]

        trigrams = []
        whole_text = []
        for abstract in self.corpus:
            whole_text = whole_text + abstract

        ngrams = Counter(nltk.ngrams(whole_text, 3))

        for ngram, freq in ngrams.most_common(11):
            trigrams.append(" ".join(ngram))
        
        self.corpus = self.corpus.apply(lambda x: " ".join(x))
        self.corpus = self.corpus.apply(lambda x: self.replace_ngram(x, bigrams))
        self.corpus = self.corpus.apply(lambda x: self.replace_ngram(x, trigrams))
        self.corpus = self.corpus.apply(lambda x: x.split())

    def find_remove_words(self):
        number_of_docs =  len(self.data.columns)
        upper_percent = 50
        lower_percent = 2

        for word in self.words_frequences.keys():
            if self.words_frequences[word] >= number_of_docs * upper_percent / 100 or self.words_frequences[word] <= number_of_docs * lower_percent / 100:
                self.remove_words.add(word)
                
    def advance_text_cleaning(self):
        self.apply_basic_cleaning()
        self.find_bigrams_trigrams()
        self.form_vocab()
        self.build_data()
        self.calc_frequences()
        self.find_remove_words()

        for i in range(len(self.corpus)):
            for j in self.remove_words:
                self.corpus[i] = list(filter(lambda a: a != j, self.corpus[i]))
        print('Cleaning was done successfully!')
    
    def simple_text_cleaning(self):
        self.apply_basic_cleaning()
        self.form_vocab()
        self.build_data()
        self.calc_frequences()
        self.find_remove_words()

        for i in range(len(self.corpus)):
            for j in self.remove_words:
                self.corpus[i] = list(filter(lambda a: a != j, self.corpus[i]))
        print('Cleaning was done successfully!')