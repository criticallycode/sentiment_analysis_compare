import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

# create a voting classifier class, will inherit from nltk's classifier

pos_text = open("positive.txt","r", encoding='utf-8', errors='replace').read()
neg_text = open("negative.txt","r", encoding='utf-8', errors='replace').read()

# create lists to store both words and reviews
# in this case the reviews are merely the individual "reviews" in the text files

all_words = []
reviews = []

# specify allowed word types form NLTK
# may or may not want to include nouns
allowed_word_types = ["J", "R", "V", "N"]
stop_words = set(stopwords.words('english'))

for p in pos_text.split('\n'):
    reviews.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            if w[0] not in stop_words:
                all_words.append(w[0].lower())

for p in neg_text.split('\n'):
    reviews.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            if w[0] not in stop_words:
                all_words.append(w[0].lower())

save_review_files = open("pickled_docs.pickle", "wb")
pickle.dump(reviews, save_review_files)
save_review_files.close()

# frequency distribution gives words in order of most commmon to least common, essentially a key:val pair with
# a frequency val for every word(key)
word_dist = nltk.FreqDist(all_words)
print(word_dist.most_common(20))

word_features = list(word_dist.keys())[:5000]
print(len(word_features))

save_features = open("pickled_features5k.pickle","wb")
pickle.dump(word_features, save_features)
save_features.close()

# now that we have the list of features, we need a way to find the features within one of the document we
# are training on
def feature_extractor(document):
    # remember that the words are the first part of the tuple, with the occurrence being the second part
    # making a set out of the doct makes it easier to search through, since it reduces it down to
    # just one representative sample for every unique element
    words = word_tokenize(document)
    features = {}
    # says that for every word in the list of word features (the words we care about)
    # the key in the features dictionary must be equal to boolean value of w in words
    # if the word in the dictionary is in the set of document (is within the document at all)
    # a True value is returned
    # NLTK will only focus on the values with True in them
    for w in word_features:
        features[w] = (w in words)
    return features

# have to convert words in every indiviudal document into features now,
# along with the label (category) for those features

# use the function we just created, check to see if the features are in that document
# returns the dictionary full of the set(unique words) of all the words we care about (word features)
# and a true or false for the values for those words, indicating whether or not they are in the document

# also get the category as the other half of the dictionary
features_sets = [(feature_extractor(review), category) for (review, category) in reviews]
random.shuffle(features_sets)

save_features_labels = open("pickled_features_labels_5k.pickle","wb")
pickle.dump(features_sets, save_features_labels)
save_features_labels.close()

# time to define training and testing set
# training is first 1900 feature sets - each individual set/instance/example is
# full of features we want (our top 3000 words) and whether or not they are in the document
# there is also the label, positive or negative
# so it can see in total how many times the words appear in negative reviews and positive reviews
training_data = features_sets[:10000]
testing_data = features_sets[10000:]

NB_clf = nltk.NaiveBayesClassifier.train(testing_data)
MNB_clf = SklearnClassifier(MultinomialNB())
BNB_clf = SklearnClassifier(BernoulliNB())
LogReg_clf = SklearnClassifier(LogisticRegression())
SGDC_clf= SklearnClassifier(SGDClassifier())
LinSVC_clf = SklearnClassifier(LinearSVC())
NuSVC_clf = SklearnClassifier(NuSVC())
KNN_clf = SklearnClassifier(KNeighborsClassifier(n_neighbors=3))
DT_clf = SklearnClassifier(DecisionTreeClassifier())

clf_list = [NB_clf, MNB_clf, BNB_clf, LogReg_clf, SGDC_clf, LinSVC_clf, NuSVC_clf, KNN_clf, DT_clf]

for clf in clf_list:
    print("Training: " + str(clf))
    clf.train(training_data)

save_nb_classifier = open("vanillanaivebayes5k.pickle","wb")
pickle.dump(NB_clf, save_nb_classifier)
save_nb_classifier.close()

save_BNB_classifier = open("BNBclf5k.pickle","wb")
pickle.dump(BNB_clf, save_BNB_classifier)
save_BNB_classifier.close()

save_MNB_classifier = open("multinaivebayes5k.pickle","wb")
pickle.dump(MNB_clf, save_MNB_classifier)
save_MNB_classifier.close()

save_LogReg_classifier = open("LogReg5k.pickle","wb")
pickle.dump(LogReg_clf, save_LogReg_classifier)
save_LogReg_classifier.close()

save_SGDC_classifier = open("SGDC5k.pickle","wb")
pickle.dump(SGDC_clf, save_SGDC_classifier)
save_SGDC_classifier.close()

save_LinSVC_classifier = open("LinSVC5k.pickle","wb")
pickle.dump(LinSVC_clf, save_LinSVC_classifier)
save_LinSVC_classifier.close()

save_NuSVC_classifier = open("NuSVC5k.pickle","wb")
pickle.dump(NuSVC_clf, save_NuSVC_classifier)
save_NuSVC_classifier.close()

save_KNN_classifier = open("KNN5k.pickle","wb")
pickle.dump(KNN_clf, save_KNN_classifier)
save_KNN_classifier.close()

save_DT_classifier = open("DT5k.pickle","wb")
pickle.dump(DT_clf, save_DT_classifier)
save_DT_classifier.close()

clf_names = {'Vanilla Naive Bayes': NB_clf,
            'Multinomial Naive Bayes': MNB_clf,
            'Bernoulli Naive Bayes': BNB_clf,
            'Logistic Regression': LogReg_clf,
            'SGDC': SGDC_clf,
            'Linear SVC': LinSVC_clf,
            'Nu SVC': NuSVC_clf,
            'Decision Tree': DT_clf,
            'K-Nearest Neighbors': KNN_clf}

for key, val in clf_names.items():
        print(key + " " + "accuracy is:")
        print(nltk.classify.accuracy(val, testing_data) * 100)

class VotingClassifier(ClassifierI):

    def __init__(self, *classifiers):
        # the classifiers of this class are the ones specified
        self.__classifiers = classifiers

    # define a classify function that overrides default classifier
    def classify(self, features):
        # stores votes from individual classifiers
        votes = []

        for i in self.__classifiers:
            v = i.classify(features)
            votes.append(v)

        # return the mode of the all the votes
        return mode(votes)

    # we may also want a confidence statistic
    # works very similar to classify but divides by total votes to get
    # confidence

    def confidence(self, features):
        votes = []
        for c in self.__classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

print("Voting classifier test:")

vote_clf = VotingClassifier(MNB_clf, BNB_clf, LogReg_clf, LinSVC_clf, NuSVC_clf, SGDC_clf, DT_clf)
print("Voted Classifier accuracy:", (nltk.classify.accuracy(vote_clf, testing_data)) * 100)

# check how its doing on a couple of examples
print("Classification:", vote_clf.classify(testing_data[0][0]),
      "Confidence: %:", vote_clf.confidence(testing_data[0][0]))

print("Classification:", vote_clf.classify(testing_data[1][0]),
      "Confidence: %:", vote_clf.confidence(testing_data[1][0]))