import pickle
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


class Classifier_():

    def __init__(self, file: str):
        self.file = file

    def code_class_labels(self):
        with open(self.file, 'rb') as f:
            dataset = pickle.load(f)
        class_labels = []
        for k in dataset.keys():
            if 'neg' in k:
                class_labels.append(0)
            elif 'pos' in k:
                class_labels.append(1)
        return dataset, class_labels

    def split_to_train_and_test(self, dataset: dict, class_labels: list):
        x_train, x_test, y_train, y_test = train_test_split(list(dataset.values()), class_labels,
                                                            train_size=0.8, stratify=class_labels)
        return x_train, x_test, y_train, y_test

    def svm_classify(self, train_x, test_x, train_y, test_y):
        param_grid = [{'kernel': ['poly'], 'degree':[n for n in range(2, 5)], 'gamma':[0.0001, 0.001, 0.01, 0.1],
                       'C':[0.1, 0.2, 1.0, 5.0, 10.0], 'coef0':[i for i in range(0, 11)], "probability":[True]},
                      {'kernel': ['rbf'], 'gamma': [0.0001, 0.001, 0.01, 0.1],
                       'C': [0.1, 0.2, 1.0, 5.0, 10.0], "probability":[True]},
                      {'kernel': ['sigmoid'], 'gamma': [0.0001, 0.001, 0.01, 0.1],
                       'C': [0.1, 0.2, 1.0, 5.0, 10.0], 'coef0': [i for i in range(0, 11)], "probability":[True]},
                      {'kernel': ['linear'], 'C':[0.1, 0.2, 1.0, 5.0, 10.0], "probability":[True]}]

        s_v_m = svm.SVC()
        clf = GridSearchCV(s_v_m, param_grid, cv=4, scoring='accuracy')
        begin = datetime.now()
        clf.fit(train_x, train_y)
        best_svm = clf.best_estimator_
        y_pred = best_svm.predict(test_x)
        punct = datetime.now()
        total = punct - begin
        print("SVM working results", classification_report(test_y, y_pred), "Working time: ", total.seconds)
        return best_svm, total.seconds

    def lr_classify(self, train_x, train_y, test_x, test_y):
        par_grid = [{'penalty': ['l1'], 'C': [0.1, 0.2, 1.0, 5.0, 10.0]},
                    {'penalty': ['l2'], 'C': [0.1, 0.2, 1.0, 5.0, 10.0]},
                    {'penalty': ['elasticnet'], 'C': [0.1, 0.2, 1.0, 5.0, 10.0], 'l1_ratio':[0.25, 0.5, 0.75]}]
        lr = LogisticRegression(solver='saga', max_iter=10000, multi_class='multinomial')
        cl = GridSearchCV(lr, par_grid, cv=4, scoring='accuracy')
        start = datetime.now()
        cl.fit(train_x, train_y)
        best_lr = cl.best_estimator_
        y_pred = best_lr.predict(test_x)
        finish = datetime.now()
        t = finish - start
        print("LR working results", classification_report(test_y, y_pred), "Working time: ", t.seconds)
        return best_lr, t.seconds

    def rf_classify(self, train_x, train_y, test_x, test_y):
        parameter_grid = {'criterion': ['gini'], 'n_estimators': [i for i in range(100, 1050, 50)],
                      "max_features": [j for j in (10, 90, 10)]}
        rf = RandomForestClassifier()
        clsf = GridSearchCV(rf, parameter_grid, cv=4, scoring='accuracy')
        s = datetime.now()
        clsf.fit(train_x, train_y)
        best_forest = clsf.best_estimator_
        y_pred = best_forest.predict(test_x)
        f = datetime.now()
        tim = f - s
        print("RF working results", classification_report(test_y, y_pred), "Working_time: ", tim.seconds)
        return best_forest, tim.seconds

    def voting_classify(self, train_x, train_y, test_x, test_y):
        cl1 = self.rf_classify(train_x, train_y, test_x, test_y)[0]
        cl2 = self.lr_classify(train_x, train_y, test_x, test_y)[0]
        cl3 = self.svm_classify(train_x, train_y, test_x, test_y)[0]
        vot = VotingClassifier(estimators=[('rf', cl1), ('lr', cl2), ('svm', cl3)], voting='soft')
        begin = datetime.now()
        vot.fit(train_x, train_y)
        y_predicted = vot.predict(test_x)
        end = datetime.now()
        work_time = end - begin
        print("Voting classifier working results", classification_report(test_y, y_predicted, output_dict=True),
              "Working_time: ", work_time.seconds)




