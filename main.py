from vectorizer import Vectorizer_
from classifier import Classifier_

path_to_data = ''  #путь к папке YELP_dataset ЧЕРЕЗ\\
my_vectorizer = Vectorizer_(path_to_data)
my_vectorizer.fill_dictionary()
my_vectorizer.doc2vec_dm_vectorize()
my_vectorizer.doc2vec_dbow_vectorize()
my_vectorizer.hashing_vectorize()
my_vectorizer.tfidf_vectorize()
my_vectorizer.tfidf_lsa_vectorize()
my_vectorizer.count_lsa_vectorize()

pickle_files = ['doc2vec_dm.pickle', 'doc2vec_dbow.pickle', 'hashing.pickle',
                'tfidf_lsa.pickle', 'count_lsa.pickle']
for file in pickle_files:
    my_classifier = Classifier_(file)
    dataset = my_classifier.code_class_labels()[0]
    class_labels = my_classifier.code_class_labels()[1]
    train_x, test_x, train_y, test_y = my_classifier.split_to_train_and_test(dataset, class_labels)
    my_classifier.voting_classify(train_x, test_x, train_y, test_y)
