from feature_extractor import FeatureExtractor
from classical_classifiers import NBayesClassifier, LogisticRegressionClassifier, XGBoostClassifier, GBoostClassifier
import pickle

def main():
    with open('extractor.pkl', 'rb') as extractor_file:
        extractor = pickle.load(extractor_file)
    with open('classifier.pkl', 'rb') as classifier_file:
        classifier = pickle.load(classifier_file)

    question = "what are the names of the states of america?"
    features = extractor.pair_question(question)
    logits = classifier.classifier.model.predict_proba(features)[:,1]
    similarities_count = len(logits[logits>0.5])
    match_indices = (-logits).argsort()[:5]

    for i in match_indices:
        print(extractor.questions[i])
    
    print('this question has', similarities_count, 'similarities')




if __name__ == '__main__':
    main()   