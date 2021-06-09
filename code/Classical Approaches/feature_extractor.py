import scipy.sparse as sp
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
class FeatureExtractor:
    def __init__(self, table, do_data = False):
        self.table = table.dropna()

        # save unique questions
        self.questions = np.unique(pd.Series(self.table['question1'].tolist() + self.table['question2'].tolist()))

        # extract features
        self.extract(self.table, do_data)

        # # visualize
        # pca = PCA(n_components=2)
        # X_2d = pca.fit_transform(self.X)
        # X_2d_pos = X_2d[self.y == 1]
        # X_2d_neg = X_2d[self.y == 0]
        # plt.scatter(*zip(*X_2d_pos), alpha=0.5)
        # plt.scatter(*zip(*X_2d_neg), alpha=0.5)
        # plt.show()

        
    def extract(self, table, do_data):
        self.stops = set(stopwords.words("english"))
        self.X = pd.DataFrame()


        # self.X['tfidf_word_match'] = self.tfidf(table)
        # if do_data:
        #     self.plot_feature(self.X['tfidf_word_match'], 'tfidf_word_match')

        self.X['word_match'] = table.apply(self.word_match_share, axis=1, raw=True)
        if do_data:
            self.plot_feature(self.X['word_match'], 'word_match')

        # self.X['jaccard'] = table.apply(self.jaccard, axis=1, raw=True)
        # if do_data:
        #     self.plot_feature(self.X['jaccard'], 'jaccard')

        self.X['wc_diff'] = table.apply(self.wc_diff, axis=1, raw=True)
        if do_data:
            self.plot_feature(self.X['wc_diff'], 'wc_diff')

        # self.X['common_words'] = table.apply(self.common_words, axis=1, raw=True)
        # if do_data:
        #     self.plot_feature(self.X['common_words'], 'common_words')

        self.X['total_unique_words'] = table.apply(self.total_unique_words, axis=1, raw=True)
        if do_data:
            self.plot_feature(self.X['total_unique_words'], 'total_unique_words')

        self.X['wc_ratio'] = table.apply(self.wc_ratio, axis=1, raw=True)
        if do_data:
            self.plot_feature(self.X['wc_ratio'], 'wc_ratio')

        self.X['wc_diff_unique'] = table.apply(self.wc_diff_unique, axis=1, raw=True)
        if do_data:
            self.plot_feature(self.X['wc_diff_unique'], 'wc_diff_unique')

        self.X['wc_ratio_unique'] = table.apply(self.wc_ratio_unique, axis=1, raw=True)
        if do_data:       
            self.plot_feature(self.X['wc_ratio_unique'], 'wc_ratio_unique')

        self.X['wc_diff_unique_stop'] = table.apply(self.wc_diff_unique_stop, axis=1, raw=True)
        if do_data:
            self.plot_feature(self.X['wc_diff_unique_stop'], 'wc_diff_unique_stop')

        # self.X['wc_ratio_unique_stop'] = table.apply(self.wc_ratio_unique_stop, axis=1, raw=True)
        # if do_data:
        #     self.plot_feature(self.X['wc_ratio_unique_stop'], 'wc_ratio_unique_stop')

        self.X['same_start_word'] = table.apply(self.same_start_word, axis=1, raw=True)
        if do_data:
            self.plot_feature(self.X['same_start_word'], 'same_start_word')

        self.X['char_diff'] = table.apply(self.char_diff, axis=1, raw=True)
        if do_data:
            self.plot_feature(self.X['char_diff'], 'char_diff')

        self.X['char_ratio'] = table.apply(self.char_ratio, axis=1, raw=True)
        if do_data:
            self.plot_feature(self.X['char_ratio'], 'char_ratio')


        if do_data:
            lbl_enc = preprocessing.LabelEncoder()
            self.y = lbl_enc.fit_transform(table.is_duplicate.values)
            self.X = self.X.fillna(0)
            # balance the data
            oversample = SMOTE()
            self.X, self.y = oversample.fit_resample(self.X, self.y)  
            self.xtrain, self.xvalid, self.ytrain, self.yvalid = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
            print(self.xvalid)
            print(type(self.xvalid))
        else:
            return self.X


 
    def word_match_share(self, row):
        q1words = {}
        q2words = {}
        for word in str(row[3]).lower().split():
            if word not in self.stops:
                q1words[word] = 1
        for word in str(row[4]).lower().split():
            if word not in self.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # questions that are nothing but stopwords
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
        return R


    def get_weight(self, count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)


    def tfidf_word_match_share(self, row):
        q1words = {}
        q2words = {}
        for word in str(row[3]).lower().split():
            if word not in self.stops:
                q1words[word] = 1
        for word in str(row[4]).lower().split():
            if word not in self.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # questions that are nothing but stopwords
            return 0
        
        shared_weights = [self.weights.get(w, 0) for w in q1words.keys() if w in q2words] + [self.weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [self.weights.get(w, 0) for w in q1words] + [self.weights.get(w, 0) for w in q2words]
        
        R = np.sum(shared_weights) / np.sum(total_weights)
        return R


    def tfidf (self, table):
        train_qs = pd.Series(table['question1'].tolist() + table['question2'].tolist()).astype(str)
        eps = 5000 
        words = (" ".join(train_qs)).lower().split()
        counts = Counter(words)
        self.weights = {word: self.get_weight(count) for word, count in counts.items()}
        tfidf_train_word_match = table.apply(self.tfidf_word_match_share, axis=1, raw=True)
        return tfidf_train_word_match

    def jaccard(self, row):
        wic = set(row[3]).intersection(set(row[4]))
        wic = [x for x in wic if x not in self.stops]
        uw = set(row[4]).union(row[3])
        uw = [x for x in uw if x not in self.stops]
        if len(uw) == 0:
            uw = [1]
        return (len(wic) / len(uw))

    def wc_diff(self, row):
        return abs(len(row[3]) - len(row[4]))

    def common_words(self, row):
        return len(set(row[3]).intersection(set(row[4])))

    def total_unique_words(self, row):
        return len(set(row[3]).union(row[4]))

    def total_unq_words_stop(self, row):
        return len([x for x in set(row[3]).union(row[4]) if x not in self.stops])


    def wc_ratio(self, row):
        l1 = len(row[3])*1.0 
        l2 = len(row[4])
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def wc_diff_unique(self, row):
        return abs(len(set(row[3])) - len(set(row[4])))

    def wc_ratio_unique(self, row):
        l1 = len(set(row[3])) * 1.0
        l2 = len(set(row[4]))
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def wc_diff_unique_stop(self, row):
        return abs(len([x for x in set(row[3]) if x not in self.stops]) - len([x for x in set(row[4]) if x not in self.stops]))

    def wc_ratio_unique_stop(self, row):
        l1 = len([x for x in set(row[3]) if x not in self.stops])*1.0 
        l2 = len([x for x in set(row[4]) if x not in self.stops])
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2

    def same_start_word(self, row):
        if not row[3] or not row[4]:
            return np.nan
        return int(row[3][0] == row[4][0])

    def char_diff(self, row):
        return abs(len(''.join(row[3])) - len(''.join(row[4])))

    def char_ratio(self, row):
        l1 = len(''.join(row[3])) 
        l2 = len(''.join(row[4]))
        if l2 == 0:
            return np.nan
        if l1 / l2:
            return l2 / l1
        else:
            return l1 / l2


    def plot_feature(self, out, feature_name):
        plt.figure(figsize=(15, 5))
        plt.hist(out[self.table['is_duplicate'] == 0].fillna(0), bins=20, density=True, label='Not Duplicate')
        plt.hist(out[self.table['is_duplicate'] == 1].fillna(0), bins=20, density=True, alpha=0.7, label='Duplicate')
        plt.legend()
        plt.title('Label distribution over ' + feature_name, fontsize=15)
        plt.xlabel(feature_name, fontsize=15)
        plt.savefig('./plots/' + feature_name + '.png')


    def pair_question(self, question):
        pairs_count = self.questions.shape[0]
        given_question_list = np.array([question] * pairs_count)
        df = pd.DataFrame(columns=self.table.columns)
        df['question1'] = given_question_list
        df['question2'] = self.questions
        df = df.fillna(0)
        features = self.extract(df, 0)
        return features
