### Sentiment Analysis Comparison

Sentiment analysis describes the use of natural language processing techniques to understand the "sentiment" or attitude of different bodies of text. Sentiment analysis can be used in the analysis of product reviews, employee feedback, trending topics and reactions, survey responses, and much more.

This project compares the output of different methods of sentiment analysis. TextBlob's  Sentiment Analysis Classifier, the VADER Sentiment Classifier, and a Custom Voting Classifier are compared here.

If you would like to see the comparison of all three sentiment analysis modules in action, there is an IPython Notebook attached to this repo.

### Custom Sentiment Analysis Module

The sentiment analysis module created here aggregates the sentiment classification of multiple different classifiers. Essentially, we'll be using multiple classifiers, having them vote on the sentiment of the text and then summarizing the votes. 

Our approach will be as follows: 

We'll prep the data for use in training our classifier, doing things like removing stop words, etc. Next we'll turn the words into features for our classifier, creating feature sets - along with their associated labels. Next, we'll train our chosen classifiers on the feature sets, and then "Pickle" the trained classifier so they don't need to be retrained.

After all the classifiers have been trained, we'll create a custom class that combines the decision of all classifiers and renders a judgement. 

After that, we'll load the pickled weights into a new file and then create a function to return the combined vote of the classifier.

First, we load in the data.

``` Python
pos_text = open("positive.txt","r", encoding='utf-8', errors='replace').read()
neg_text = open("negative.txt","r", encoding='utf-8', errors='replace').read()
```

Then we'll create lists to store all the words and the individual "documents" (for the purposes of our training set, really just lines in our training docs.)

``` Python
all_words = []
reviews = []
```

Now we need to decide what parts of speech we'll be training the classifiers on. In NLTK these are adjectives, adverbs, verbs, and nouns, respectively.

We also need to define stop words, so our algorithms don't worry about common words.

``` Python
# specify allowed word types form NLTK
# may or may not want to include nouns
allowed_word_types = ["J", "R", "V", "N"]
stop_words = set(stopwords.words('english'))
```

We now need to preprocess our text files. We'll make tokens out of them and then select just the words with parts of speech we are interested in to train on. We need to do this for both the positive and negative document.

``` Python
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

```

We then need to choose how many words we want to train on. We get a frequency distribution of all the words in our training database and then select the only the top "X" number of words to train on, to become features.

``` Python
word_dist = nltk.FreqDist(all_words)
print(word_dist.most_common(20))

word_features = list(word_dist.keys())[:5000]
print(len(word_features))
```

We now need to our select the individual training sentences/reviews in our list of word features and get both the features and the labels. We also need to shuffle the feature sets, or they will be all positive and then all negative.

``` Python
def feature_extractor(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

features_sets = [(feature_extractor(review), category) for (review, category) in reviews]
random.shuffle(features_sets)
```

Now that we have all the reviews, features, labels, and feature sets created, we need to pickle them so we can use them later.

``` Python
save_review_files = open("pickled_docs.pickle", "wb")
pickle.dump(reviews, save_review_files)
save_review_files.close()

save_features = open("pickled_features5k.pickle","wb")
pickle.dump(word_features, save_features)
save_features.close()

save_features_labels = open("pickled_features_labels_5k.pickle","wb")
pickle.dump(features_sets, save_features_labels)
save_features_labels.close()
```

We can now divide our data into training and testing sets.

```Python
training_data = features_sets[:10000]
testing_data = features_sets[10000:]
```

Now comes the time to decide what classifiers we want to use.

``` Python
NB_clf = nltk.NaiveBayesClassifier.train(testing_data)
MNB_clf = SklearnClassifier(MultinomialNB())
BNB_clf = SklearnClassifier(BernoulliNB())
LogReg_clf = SklearnClassifier(LogisticRegression())
SGDC_clf= SklearnClassifier(SGDClassifier())
LinSVC_clf = SklearnClassifier(LinearSVC())
NuSVC_clf = SklearnClassifier(NuSVC())
KNN_clf = SklearnClassifier(KNeighborsClassifier(n_neighbors=3))
DT_clf = SklearnClassifier(DecisionTreeClassifier())
```

Let's train them all on the training set.

``` Python
clf_list = [NB_clf, MNB_clf, BNB_clf, LogReg_clf, SGDC_clf, LinSVC_clf, NuSVC_clf, KNN_clf, DT_clf]

for clf in clf_list:
    print("Training: " + str(clf))
    clf.train(training_data)
```

Because the training takes so long, we'll definitely want to Pickle the resulting classifier.

```  Python
save_nb_classifier = open("vanillanaivebayes5k.pickle","wb")
pickle.dump(NB_clf, save_nb_classifier)
save_nb_classifier.close()
```

We'll do that for each of the chosen classifiers. There's nine classifiers total. We've chosen an odd number so that there's no ties in the voting. Now we just need to see how the classifiers performs on the training set.

``` Python
clf_names = {'Vanilla Naive Bayes': NB_clf,
            'Multinomial Naive Bayes': MNB_clf,
            'Bernoulli Naive Bayes': BNB_clf,
            'Logistic Regression': LogReg_clf,
            'SGDC': SGDC_clf,
            'Linear SVC': LinSVC_clf,
            'Nu SVC': NuSVC_clf,
            'KNN': KNN_clf,
            'Decision Tree': DT_clf}

for key, val in clf_names.items():
        print(key + " " + "accuracy is:")
        print(nltk.classify.accuracy(val, testing_set) * 100)
```

Here's the printouts.

```
Vanilla Naive Bayes accuracy is:
94.57831325301204
Multinomial Naive Bayes accuracy is:
70.18072289156626
Bernoulli Naive Bayes accuracy is:
72.28915662650603
Logistic Regression accuracy is:
72.43975903614458
SGDC accuracy is:
72.13855421686746
Linear SVC accuracy is:
71.6867469879518
Nu SVC accuracy is:
72.89156626506023
KNN accuracy is:
51.80722891566265
Decision Tree accuracy is:
67.16867469879519
```

Regular Naive Bayes gets suspiciously high performance, could be overfitting. Let's drop regular Naive Bayes and the other weakest performing algorithms. 

Now we'll create a class to combine votes.

``` Python
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

```

We need to create an instance of the classifier class with just the classifiers we have chosen.

``` Python
vote_clf = VotingClassifier(MNB_clf, BNB_clf, LogReg_clf, LinSVC_clf, NuSVC_clf, SGDC_clf, DT_clf)
print("Voted Classifier accuracy:", (nltk.classify.accuracy(vote_clf, testing_data)) * 100)
```

Let's also check to see how our voting classifier is performing on the training set.

``` Python
Voted Classifier accuracy: 73.49397590361446
```

Not bad, this is performing at least as well or better than our chosen classifiers. In practice, this tends to vary but the accuracy of the voted classifier always seems to be higher than the  lowest individual accuracies.  A voting classifier/ensemble method should be more robust to overfitting as well.

Now to use it, we just load the pickled data back in.

``` Python
documents_file = open("pickled_docs.pickle", "rb")
documents = pickle.load(documents_file)
documents_file.close()

word_features_file = open("pickled_features5k.pickle", "rb")
word_features = pickle.load(word_features_file)
word_features_file.close()

feature_sets_file = open("pickled_features_sets_5k.pickle", "rb")
feature_sets = pickle.load(feature_sets_file)
feature_sets_file.close()

# Do this for all the classifiers
open_clf = open("multinaivebayes5k.pickle", "rb")
MNB_clf = pickle.load(open_clf)
open_clf.close()

```

Now we create a function to get the features.

``` Python
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
```

Finally, we can create a function to use the voting classifier.

``` Python
def sentiment(text):
    feats = find_features(text)
    return vote_clf.classify(feats), vote_clf.confidence(feats)
```

Now we just have to import the function into another script and we can use it.

``` Python
import usablesentimentmod as s

text = "This game is terrible. The controls are garbage and so are the character models, I hate everything about it."
text2 = "I love her so much. She makes me really happy."

analyzer = s.sentiment(text)
analyzer2 = s.sentiment(text2)

print(analyzer)
print(analyzer2)
```

We can now compare the performance of this custom analyzer to the two premade sentiment analyzers.

``` Python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentences = [text, text2]

analyzer = SentimentIntensityAnalyzer()

for s in sentences:
    print("Sentence classification - " + "Test sentence is :  " + str(s))
    print("---")
    text_blob = TextBlob(s)
    # For Textblob, range runs from [-1.0, 1.0], Above 0 is positive
    print("TextBlob: " + str(text_blob.sentiment.polarity))
    v_sent = analyzer.polarity_scores(s)
    print("Vader: " + str(v_sent))
    custom = sentiment(s)
    print("Custom: " + str(custom))
    print()
```

Thanks to PythonProgramming.net, whose work inspired this custom sentiment analysis classifier, and who linked the data of short positive and negative reviews.