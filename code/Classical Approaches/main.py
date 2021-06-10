from feature_extractor import FeatureExtractor
from classical_classifiers import NBayesClassifier, LogisticRegressionClassifier, XGBoostClassifier, GBoostClassifier
import numpy as np
import argparse
import pandas as pd
import pickle


def main(classifier, do_data):

    # extract Features from raw data
    if do_data:
        # read the data
        table = pd.read_csv('../dataset/train.csv')

        # extract the features
        extractor = FeatureExtractor(table, do_data)
        # save the extractor
        with open('extractor.pkl', 'wb') as extractor_file:
            pickle.dump(extractor, extractor_file)

    else:
        with open('extractor.pkl', 'rb') as extractor_file:
            extractor = pickle.load(extractor_file)
 
        # classical classifiers
        # Naive Bayes classifiers 
        if classifier == 'naive_bayes':
            NBayes_classifier = NBayesClassifier(xtrain = extractor.xtrain, ytrain = extractor.ytrain)
            print('vaidation accuracy of Naive Bayes classifier = ', NBayes_classifier.classifier.get_accuracy(extractor.xvalid, extractor.yvalid))
            NBayes_classifier.classifier.plot_roc_curve(extractor.xvalid, extractor.yvalid)
            print('AUC of Naive Bayes classifier = ', NBayes_classifier.classifier.auc)
            # save the classifier
            with open('classifier.pkl', 'wb') as classifier_file:
                pickle.dump(NBayes_classifier, classifier_file)

        # Logistic Regression classifiers 
        elif classifier == 'logistic_regression':
            LogReg_classifier = LogisticRegressionClassifier(xtrain = extractor.xtrain, ytrain = extractor.ytrain)
            print('vaidation accuracy of Logistic Regression classifier = ', LogReg_classifier.classifier.get_accuracy(extractor.xvalid, extractor.yvalid))
            LogReg_classifier.classifier.plot_roc_curve(extractor.xvalid, extractor.yvalid)
            print('AUC of Logistic Regression classifier = ', LogReg_classifier.classifier.auc)
            # save the classifier
            with open('classifier.pkl', 'wb') as classifier_file:
                pickle.dump(LogReg_classifier, classifier_file)     

        # XGBOOST classifiers 
        elif classifier == 'xgboost':
            xgb_classifier = XGBoostClassifier(xtrain = extractor.xtrain, ytrain = extractor.ytrain)
            print('vaidation accuracy of XGBoost classifier = ', xgb_classifier.classifier.get_accuracy(extractor.xvalid, extractor.yvalid))
            xgb_classifier.classifier.plot_roc_curve(extractor.xvalid, extractor.yvalid)
            print('AUC of XGBoost classifier = ', xgb_classifier.classifier.auc)
            # save the classifier
            with open('classifier.pkl', 'wb') as classifier_file:
                pickle.dump(xgb_classifier, classifier_file)    

        # GBOOST classifiers 
        elif classifier == 'gboost':
            gb_classifier = GBoostClassifier(xtrain = extractor.xtrain, ytrain = extractor.ytrain)
            print('vaidation accuracy of GBoost classifier = ', gb_classifier.classifier.get_accuracy(extractor.xvalid, extractor.yvalid))
            gb_classifier.classifier.plot_roc_curve(extractor.xvalid, extractor.yvalid)
            print('AUC of GBoost classifier = ', gb_classifier.classifier.auc)
            # save the classifier
            with open('classifier.pkl', 'wb') as classifier_file:
                pickle.dump(gb_classifier, classifier_file)     

        else:
            print('unsupported classifier, choose one classifier from(logistic_regression, naive_bayes, xgboost)...')




if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-classifier', '--classifier', type=str, help='choose one classifier from(logistic_regression, naive_bayes, xgboost), default is xgboost', default = 'xgboost')
    argparser.add_argument("--do_data", action='store_true')
    args = argparser.parse_args()
    main(args.classifier, args.do_data)       
