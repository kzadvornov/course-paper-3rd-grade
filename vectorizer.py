import os
import re
import gensim
import pickle
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from datetime import datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class Vectorizer_():

    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset  #путь к папке Yelp dataset на компьютере ЧЕРЕЗ \\
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words('english')
        self.corpus = [path_to_dataset+'\\negative', path_to_dataset+'\\positive']
        self.all_docs = {}
        self.d2v_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate([v for v in self.all_docs.values()][0:601])]

    def space_division(self, text: str):
        wrong = re.findall("[a-z0-9A-Z][!?.:][A-Za-z]", text)
        corr = []
        for c in [list(el) for el in wrong]:
            c.insert(2, ' ')
        corr.append(c)
        crc = [''.join(it) for it in corr]
        for n in range(len(crc)):
            text = text.replace(wrong[n], crc[n])
        return text

    def space_inserting(self, text: str):
        fal = re.findall("[a-z][A-Z]", text)
        tru = []
        for c in [list(el) for el in fal]:
            c.insert(1," ")
            tru.append(c)
        tr = [''.join(e) for e in tru]
        for n in range(len(tr)):
            text = text.replace(fal[n], tr[n])
        return text

    def filter_punct_and_numbers(self, text: str):
        not_needed = set(re.findall("[^A-Za-z\s'-]", text))
        for sym in not_needed:
            text = text.replace(sym, ' ').lower()
        for sp in re.findall("\s{2,}", text):
            text = text.replace(sp, " ")
        return text

    def filter_stop_words(self, text: str):
        text = text.split(" ")
        text = list(filter(lambda x: x not in stop_words, text))
        return text

    def filter_empties_and_dashes(self, text):
        text = list(filter(lambda x: x not in stop_words and x != '' and x != '-' and x != '--', text))
        for i in range(len(text)):
            if re.match('^-[a-z]+$', text[i]) or re.match('^[a-z]+-$', text[i]):
               text[i] = text[i].replace('-', '')
        return text

    def final_preprocess(self, text):
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        text = " ".join(text)
        text = [word for word in word_tokenize(text) if word not in stop_words and not re.match("[a-z]?'[a-z]", word)]
        fin_text = []
        for c in pos_tag(text):
            if c[1][0].upper() in tag_dict.keys():
                fin_text.append(self.lemmatizer.lemmatize(c[0], tag_dict[c[1][0].upper()]))
            else:
                fin_text.append(c[0])
        return fin_text

    def preprocess(self, text):
        text = self.space_division(text)
        text = self.space_inserting(text)
        text = self.filter_punct_and_numbers(text)
        text = self.filter_stop_words(text)
        text = self.filter_empties_and_dashes(text)
        text = self.final_preprocess(text)
        return text

    def fill_dictionary(self):
        for c in self.corpus:
        for i in range(len(os.listdir(c))):
            self.all_docs[os.listdir(c)[i]] = self.preprocess(open(c + '\\' + os.listdir(c)[i], 'r', encoding='utf-8').read())


    def doc2vec_dm_vectorize(self):
        doc2vec_dm_vectorized = {}
        start = datetime.now()
        my_dm = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=100, min_count=1, epochs=10)
        my_dm.build_vocab(self.d2v_corpus)
        my_dm.train(self.d2v_corpus, total_examples=my_dm.corpus_count, epochs=my_dm.epochs)
        for k in self.all_docs.keys():
            doc2vec_dm_vectorized[k] = my_dm.infer_vector(self.all_docs[k])
        finish = datetime.now()
        print("Doc2Vec_DM working time:", (finish-start).seconds)
        with open('doc2vec_dm.pickle', 'wb') as file:
             pickle.dump(doc2vec_dm_vectorized, file)

    def doc2vec_dbow_vectorize(self):
        doc2vec_dbow_vectorized = {}
        start = datetime.now()
        my_dbow = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=100, min_count=1, epochs=10)
        my_dbow.build_vocab(self.d2v_corpus)
        my_dbow.train(self.d2v_corpus, total_examples=my_dbow.corpus_count, epochs=my_dbow.epochs)
        for k in self.all_docs.keys():
            doc2vec_dbow_vectorized[k] = my_dbow.infer_vector(self.all_docs[k])
        finish = datetime.now()
        print("Doc2Vec_DBOW working time:", (finish - start).seconds)
        with open('doc2vec_dbow.pickle', 'wb') as f:
            pickle.dump(doc2vec_dbow_vectorized, f)

    def hashing_vectorize(self):
        hash_vectorized = {}
        work_corp = [" ".join(v) for v in self.all_docs.values()]
        start = datetime.now()
        vectorizer = HashingVectorizer(ngram_range=(1, 3), n_features=100000)
        svd = TruncatedSVD(n_components=100)
        hv = svd.fit_transform(vectorizer.fit_transform(work_corp).toarray())
        key_list = [k for k in self.all_docs.keys()]
        for i in range(len(key_list)):
            hash_vectorized[key_list[i]] = hv[i]
        finish = datetime.now()
        print("HashingVectorizer working time:", (finish - start).seconds)
        with open('hashing.pickle', 'wb') as file:
             pickle.dump(hash_vectorized, file)

    def tfidf_vectorize(self):
        tfidf_vectorized = {}
        work_corp = [" ".join(v) for v in self.all_docs.values()]
        start = datetime.now()
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        vs = vectorizer.fit_transform(work_corp)
        matr = vs.toarray()
        key_list = [k for k in self.all_docs.keys()]
        for i in range(len(key_list)):
            tfidf_vectorized[key_list[i]] = matr[i]
        finish = datetime.now()
        print("TfidfVectorizer working time:", (finish - start).seconds)
        with open('tf_idf.pickle', 'wb') as file:
             pickle.dump(tfidf_vectorized, file)

    def tfidf_lsa_vectorize(self):
        tfidf_lsa_vectorized = {}
        with open('tf_idf.pickle', 'rb') as file:
             tfidfs = pickle.load(file)
        matrix = [v for v in tfidfs.values()]
        key_list = [k for k in tfidfs.keys()]
        start = datetime.now()
        svd = TruncatedSVD(n_components=100)
        lsa = svd.fit_transform(matrix)
        for i in range(len(key_list)):
            tfidf_lsa_vectorized[key_list[i]] = lsa[i]
        finish = datetime.now()
        print("Tf-Idf dimensionality reduction time:", (finish - start).seconds)
        with open('tfidf_lsa.pickle', 'wb') as f:
             pickle.dump(tfidf_lsa_vectorized, f)

    def count_lsa_vectorize(self):
        count_lsa_vectorized = {}
        key_list = [k for k in self.all_docs.keys()]
        corp_to_work = [" ".join(v) for v in self.all_docs.values()]
        start = datetime.now()
        vectorizer = CountVectorizer(ngram_range=(1, 3))
        arr = vectorizer.fit_transform(corp_to_work).toarray()
        svd = TruncatedSVD(n_components=100)
        lsa = svd.fit_transform(arr)
        for i in range(len(key_list)):
            count_lsa_vectorized[key_list[i]] = lsa[i]
        finish = datetime.now()
        print("LSA vectorizer working time:", (finish - start).seconds)
        with open('count_lsa.pickle', 'wb') as file:
             pickle.dump(count_lsa_vectorized, file)
