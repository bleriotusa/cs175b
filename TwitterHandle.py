__author__ = 'Michael'
import ast
import time
import glob
import nltk
import random
from nltk.corpus import stopwords
stopwordslist = stopwords.words('english')
from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterRestPager
from datetime import datetime
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SoftmaxLayer
from pybrain.utilities import percentError

consumer_key = 'XKHtqeaKlCR9n4BG3MI5cwftj'
consumer_secret = 'NcQXp5pStiFqJnPn3DjJZZyHRaNb7lV01lcOfZ4t42QfoA08pQ'
access_token_key = '168559355-Qi1ARhn8IkDls2MlFYZtzX0fXhNEkX51DEKK0ni1'
access_token_secret = 'BV7mPSXRnXel0frvjsMUh5R2ePmlDriyXj7nGI5P0nZ5d'


class Tweet:
    """
    Small class to hold the tweet information.
    """

    def __init__(self):
        self.hashtags = None
        self.text = None
        self.target = None

    def __eq__(self, other):
        return self.text == other.text

    def __hash__(self):
        return hash(self.text)


def pull_tweets(tweets: int, hashtag: str) -> None:
    """
    Pulls specified number of tweets and writes them into file. Layout is:
        odd lines - hashtag list
        even lines - text or tweet message
    :param tweets: number of tweets to pull
    :return: None
    """

    start_time = datetime.now()
    print(start_time)

    api = TwitterAPI(consumer_key, consumer_secret, access_token_key, access_token_secret)

    data_file = open('data/{}{}'.format(str(start_time), '.txt'), 'wb+')
    r = TwitterRestPager(api, 'search/tweets', {'q': '#{}'.format(hashtag), 'count': 100, 'lang': 'en'})

    tweet_set = set()
    # while len(tweet_set) < tweets:
    for item in r.get_iterator():
        tweet = Tweet()
        if len(tweet_set) >= tweets:
            break
        if 'text' in item:
            tweet.hashtags = [hashtag['text'] for hashtag in item['entities']['hashtags']]
            tweet.text = item['text'].replace('\n', ' ')
            tweet.target = hashtag
            if tweet not in tweet_set:
                tweet_set.add(tweet)
                print(tweet.hashtags, tweet.text, tweet.target)
            print(len(tweet_set))

        elif 'message' in item and item['code'] == 88:
            print('SUSPEND, RATE LIMIT EXCEEDED: %s\n' % item['message'])
            time.sleep(16 * 60)

    pickle.dump(tweet_set, data_file, 2)
    data_file.close()
    print(datetime.now() - start_time)
    return start_time


def read_data(filename: str) -> list:
    """
    :param filename: file to read from
    :return: list of Tweet objects
    """

    with open(filename, 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError:
            return set()


def read_all_data(tone=None):
    all_data = set()
    for filename in glob.glob("data/*.txt"):
        data = read_data(filename)
        all_data.update(data)

    # print(all_data)

    target_data = {tweet for tweet in all_data if tweet.target == tone} if tone else all_data

    import re
    if tone:
        tone_replace = re.compile(re.escape(tone), re.IGNORECASE)
        for tweet in target_data:
            tweet.text = tone_replace.sub('', tweet.text)
    # for tweet in target_data:
    #     print('{}: {}\n\t{}'.format(tweet.target, tweet.hashtags, tweet.text))

    return target_data

# Remove stop words
from sklearn.feature_extraction.text import CountVectorizer


# Override default countvectorizer tokenizer to remove stopwords
def remove_stop_word_tokenizer(s):
    count_vect = CountVectorizer()
    default_tokenizer_function = count_vect.build_tokenizer()
    words = default_tokenizer_function(s)
    words = list(w for w in words if w.lower() not in stopwordslist)
    return words


count_vect = CountVectorizer()
count_vect.tokenizer = remove_stop_word_tokenizer
tfidf_transformer = TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)

def trainNaiveBayes(data: list, targets: list):
    X_tweet_counts = count_vect.fit_transform(data)

    # Compute term frequencies and store in X_train_tf
    # Compute tfidf feature values and store in X_train_tfidf
    X_train_tfidf = tfidf_transformer.fit_transform(X_tweet_counts)

    # train and test a Multinomial Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    return MultinomialNB().fit(X_train_tfidf, targets)

def trainLogisticRegression(data: list, targets: list):
    X_tweet_counts = count_vect.fit_transform(data)

    # Compute term frequencies and store in X_train_tf
    # Compute tfidf feature values and store in X_train_tfidf
    X_train_tfidf = tfidf_transformer.fit_transform(X_tweet_counts)

    # train and test a Multinomial Naive Bayes Classifier
    from sklearn.linear_model.logistic import LogisticRegression
    return LogisticRegression().fit(X_train_tfidf, targets)

def trainSVM(data: list, targets: list):
    X_tweet_counts = count_vect.fit_transform(data)

    # Compute term frequencies and store in X_train_tf
    # Compute tfidf feature values and store in X_train_tfidf
    X_train_tfidf = tfidf_transformer.fit_transform(X_tweet_counts)

    # train and test a Multinomial Naive Bayes Classifier
    from sklearn.svm import SVC
    return SVC().fit(X_train_tfidf, targets)

def predict(predictor, test_data):
    # count_vect = CountVectorizer()
    # count_vect.tokenizer = remove_stop_word_tokenizer

    X_new_counts = count_vect.transform(test_data)
    # tfidf_transformer = TfidfTransformer()
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predictedNB = predictor.predict(X_new_tfidf)

    # for doc, category in zip(test_data, predictedNB):
    #      print('%r %s' % (doc, category))
    # print('\n')

    return predictedNB

def test(positive: list, negative: list, seed: int, trainingFunction):
    all_data = positive + negative

    # creates a list of target values. Positive entries will be "1" and negative entries will be "0"
    targets = [1] * len(positive)
    targets = targets + ([0] * len(negative))

    random.seed(seed)
    random.shuffle(all_data)
    random.seed(seed)
    random.shuffle(targets)

    training_data = all_data[:int(.75 * len(all_data))]
    test_data = all_data[int(.75 * len(all_data)):]

    training_targets = targets[:int(.75 * len(targets))]
    test_targets = targets[int(.75 * len(targets)):]

    predictor = None
    if trainingFunction is None:
        predictor = trainNaiveBayes(training_data, training_targets)
    else:
        predictor = trainingFunction(training_data, training_targets)
    predicted = predict(predictor, test_data)

    for text, prediction, target in zip(test_data, predicted, test_targets):
        print(prediction, target, text)

    successes = [1 for prediction, target in zip(predicted, test_targets) if prediction == target]
    print (len(successes) / len(test_data))
    return predictor


def testNN(positive: list, negative: list, seed: int):
    all_data = positive + negative

    # creates a list of target values. Positive entries will be "1" and negative entries will be "0"
    targets = [1] * len(positive)
    targets = targets + ([0] * len(negative))

    random.seed(seed)
    random.shuffle(targets)
    random.seed(seed)
    random.shuffle(all_data)


    predictor = trainNN(all_data, targets, seed)
    # predicted = predictNN(predictor, test_data)

    # for doc, category in zip(test_data, predicted):
    #      print('%r %s' % (doc, category))
    # print('\n')

    return predictor


def trainNN(data: list, targets: list, seed):
    X_tweet_counts = count_vect.fit_transform(data)

    # Compute term frequencies and store in X_train_tf
    # Compute tfidf feature values and store in X_train_tfidf
    X_train_tfidf = tfidf_transformer.fit_transform(X_tweet_counts)
    arr = X_train_tfidf.toarray()

    trainingdata = arr[:int(.75 * len(arr))]
    testdata = arr[int(.75 * len(arr)):]
    trainingtargets = targets[:int(.75 * len(targets))]
    testtargets = targets[int(.75 * len(targets)):]


    trainingds = ClassificationDataSet(len(arr[0]), 1, nb_classes=2)
    testds = ClassificationDataSet(len(arr[0]), 1, nb_classes=2)


    for index, data in enumerate(trainingdata):
        trainingds.addSample(data, trainingtargets[index])
    for index, data in enumerate(testdata):
        testds.addSample(data, testtargets[index])

    trainingds._convertToOneOfMany()
    testds._convertToOneOfMany()

    net = buildNetwork( trainingds.indim, 10, 10, 10, trainingds.outdim, outclass=SoftmaxLayer )
    trainer = BackpropTrainer(net, dataset=trainingds, learningrate=.75, momentum=.1)

    for i in range(25):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                 trainingds['class'])
        tstresult = percentError(trainer.testOnClassData(
                                 dataset=testds), testds['class'])

        print("epoch: %4d" % trainer.totalepochs,
                     "  train error: %5.2f%%" % trnresult,
                     "  test error: %5.2f%%" % tstresult)

    return net

def predictNN(predictor, test_data):
    # count_vect = CountVectorizer()
    # count_vect.tokenizer = remove_stop_word_tokenizer

    X_new_counts = count_vect.transform(test_data)
    # tfidf_transformer = TfidfTransformer()
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    arr = X_new_tfidf.toarray()
    results = []
    for index, data in enumerate(arr):
        predictedNB = predictor.activate(data)
        print(predictedNB)

    # for doc, category in zip(test_data, predictedNB):
    #      print('%r %s' % (doc, category))
    # print('\n')

    return results

if __name__ == '__main__':
    #
    # pull_tweets(5000, 'courage')
    # pull_tweets(5000, 'scared')
    # pull_tweets(5000, 'sarcasm')
    # pull_tweets(5000, 'serious')
    # pull_tweets(5000, 'relaxed')
    # pull_tweets(5000, 'stressed')

    # happy = list(read_all_data('happy'))
    # sad = list(read_all_data('sad'))
    # fearful = list(read_all_data('scary'))
    # courageous = list(read_all_data('courage'))
    sarcastic = list(read_all_data('sarcasm'))
    # sincere = list(read_all_data('serious'))
    # relaxed = list(read_all_data('relaxed'))
    stressed = list(read_all_data('stressed'))


    # Ensures that the length of the two datasets are the same, so that there's a 50% chance of being right by default
    # happylen = min(len(happy), len(sad))
    # couragelen = min(len(fearful), len(courageous))
    sarcasmlen = min(len(sarcastic), len(stressed))
    # relaxedlen = min(len(relaxed), len(stressed))

    #
    # print(len(happy), len(sad), len(fearful), len(courageous),
    #       len(sarcastic), len(sincere), len(relaxed), len(stressed))

    results = []
    # results = [test([w.text for w in happy[:happylen]], [w.text for w in sad[:happylen]], 1)]
    # results.append(test([w.text for w in courageous[:couragelen]], [w.text for w in fearful[:couragelen]], 1))
    # results.append(test([w.text for w in sarcastic[:sarcasmlen]], [w.text for w in stressed[:sarcasmlen]], 1))
    # results.append(test([w.text for w in stressed[:relaxedlen]], [w.text for w in relaxed[:relaxedlen]], 1))
    testNN([w.text for w in sarcastic[:sarcasmlen]], [w.text for w in stressed[:sarcasmlen]], 1)
    # print(results)

    # results = []
    # predictorhappy = test([w.text for w in happy[:happylen]], [w.text for w in sad[:happylen]], 1)
    # results.append(predict(predictorhappy, ["Birthday"]))
    # predictorconfidence = test([w.text for w in courageous[:couragelen]], [w.text for w in fearful[:couragelen]], 1)
    # results.append(predict(predictorconfidence, ["Birthday"]))
    # predictorsarcasm = test([w.text for w in sarcastic[:sarcasmlen]], [w.text for w in sincere[:sarcasmlen]], 1)
    # results.append(predict(predictorsarcasm, ["Birthday"]))
    # predictorrelaxed = test([w.text for w in stressed[:relaxedlen]], [w.text for w in relaxed[:relaxedlen]], 1)
    # results.append(predict(predictorrelaxed, ["Birthday"]))
    # print(results)