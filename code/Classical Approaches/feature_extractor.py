import scipy.sparse as sp
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

class FeatureExtractor:
    def __init__(self, table, do_data = False):
        self.extract(table, do_data)

        
    def extract(self, table, do_data):
        self.stops = set(stopwords.words("english"))
        self.table = table.dropna()
        self.X = pd.DataFrame()
        # self.X['tfidf_word_match'] = self.tfidf()
        self.X['word_match'] = self.table.apply(self.word_match_share, axis=1, raw=True)
        # self.X['jaccard'] = self.table.apply(self.jaccard, axis=1, raw=True)
        self.X['wc_diff'] = self.table.apply(self.wc_diff, axis=1, raw=True)
        # self.X['common_words'] = self.table.apply(self.common_words, axis=1, raw=True)
        self.X['total_unique_words'] = self.table.apply(self.total_unique_words, axis=1, raw=True)
        self.X['wc_ratio'] = self.table.apply(self.wc_ratio, axis=1, raw=True)
        self.X['wc_diff_unique'] = self.table.apply(self.wc_diff_unique, axis=1, raw=True)
        self.X['wc_ratio_unique'] = self.table.apply(self.wc_ratio_unique, axis=1, raw=True)
        self.X['wc_diff_unique_stop'] = self.table.apply(self.wc_diff_unique_stop, axis=1, raw=True)
        # self.X['wc_ratio_unique_stop'] = self.table.apply(self.wc_ratio_unique_stop, axis=1, raw=True)
        self.X['same_start_word'] = self.table.apply(self.same_start_word, axis=1, raw=True)
        self.X['char_diff'] = self.table.apply(self.char_diff, axis=1, raw=True)
        self.X['char_ratio'] = self.table.apply(self.char_ratio, axis=1, raw=True)

        if do_data:
            lbl_enc = preprocessing.LabelEncoder()
            self.y = lbl_enc.fit_transform(self.table.is_duplicate.values)
            self.X = self.X.fillna(0)
            # balance the data
            oversample = SMOTE()
            self.X, self.y = oversample.fit_resample(self.X, self.y)  
            self.xtrain, self.xvalid, self.ytrain, self.yvalid = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        
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
        for word in str(row['question1']).lower().split():
            if word not in self.stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in self.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # questions that are nothing but stopwords
            return 0
        shared_weights = [self.weights.get(w, 0) for w in q1words.keys() if w in q2words] + [self.weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [self.weights.get(w, 0) for w in q1words] + [self.weights.get(w, 0) for w in q2words]
        R = np.sum(shared_weights) / np.sum(total_weights)
        return R


    def tfidf (self):
        train_qs = list(pd.Series(self.table['question1'].tolist() + self.table['question2'].tolist()).astype(str))
        eps = 5000 
        words = (" ".join(train_qs)).lower().split()
        counts = Counter(words)
        self.weights = {word: self.get_weight(count) for word, count in counts.items()}
        tfidf_train_word_match = []
        for index, row in self.table.iterrows():
            tfidf_train_word_match.append(self.tfidf_word_match_share(row))
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


