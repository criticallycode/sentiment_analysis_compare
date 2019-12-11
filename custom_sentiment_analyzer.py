from nltk.tokenize import word_tokenize
import pickle
from nltk.classify import ClassifierI
from statistics import mode

# this custom classifier inherits from nltk.classify class
class VotingClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self.__classifiers = classifiers

    # need to override default classify function
    def classify(self, features):
        # create an empty list to hold the votes from all the different classifiers
        votes = []
        # for the classification from every classifier - append classification to votes
        for c in self.__classifiers:
            v = c.classify(features)
            votes.append(v)
        # return the mode of the votes
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self.__classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf

word_features_file = open("pickled_features5k.pickle", "rb")
word_features = pickle.load(word_features_file)
word_features_file.close()

def find_features(document):
    # remember that the words are the first part of the tuple, with the label being the second part
    # making a set out of the doct makes it easier to search through, since it reduces it down to
    # just one representative sample for every unique element
    words = word_tokenize(document)
    features = {}
    # says that for every word in the list of word features (the words we care about)
    # the key in the features dictionary must be equal to boolean value of w in words
    # if the word in the dictionary is in the set of document (is within the document at all)
    # a True value is returned
    for w in word_features:
        features[w] = (w in words)
    return features

open_clf = open("multinaivebayes5k.pickle", "rb")
MNB_clf = pickle.load(open_clf)
open_clf.close()

open_clf = open("BNBclf5k.pickle", "rb")
BNB_clf = pickle.load(open_clf)
open_clf.close()

open_clf = open("LogReg5k.pickle", "rb")
LogReg_clf = pickle.load(open_clf)
open_clf.close()

open_clf = open("LinSVC5k.pickle", "rb")
LinSVC_clf = pickle.load(open_clf)
open_clf.close()

# open_clf = open("NuSVC5k.pickle", "rb")
# NuSVC_clf = pickle.load(open_clf)
# open_clf.close()

open_clf = open("SGDC5k.pickle", "rb")
SGDC_clf = pickle.load(open_clf)
open_clf.close()

# open_clf = open("DT5k.pickle", "rb")
# DT_clf = pickle.load(open_clf)
# open_clf.close()

vote_clf = VotingClassifier(MNB_clf, BNB_clf, LogReg_clf, LinSVC_clf, SGDC_clf)

# just import this into another script and run

def sentiment(text):
    feats = find_features(text)
    return vote_clf.classify(feats), vote_clf.confidence(feats)

