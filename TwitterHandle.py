__author__ = 'Michael'
import time
import glob
import random
from nltk.corpus import stopwords

stopwordslist = stopwords.words('english')
from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterRestPager
from datetime import datetime
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
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

    # use this so that we don't retrieve tweets that we have already gotten
    r = TwitterRestPager(api, 'search/tweets', {'q': '#{}'.format(hashtag), 'count': 100, 'lang': 'en'})

    tweet_set = set()
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


all_data = set()


def read_all_data(tone=None):
    global all_data

    # retrieve all data from stored files once
    if not all_data:
        all_data = set()
        for filename in glob.glob("data/*.txt"):
            data = read_data(filename)
            all_data.update(data)

    target_data = {tweet for tweet in all_data if tweet.target == tone} if tone else all_data

    import re

    if tone:
        tone_replace = re.compile(re.escape(tone), re.IGNORECASE)
        for tweet in target_data:
            tweet.text = tone_replace.sub('', tweet.text)

    return target_data

# Remove stop words
from sklearn.feature_extraction.text import CountVectorizer


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

    # train and test a Logistic Regression Classifier
    from sklearn.linear_model.logistic import LogisticRegression

    return LogisticRegression().fit(X_train_tfidf, targets)


def trainSVM(data: list, targets: list):
    X_tweet_counts = count_vect.fit_transform(data)

    # Compute term frequencies and store in X_train_tf
    # Compute tfidf feature values and store in X_train_tfidf
    X_train_tfidf = tfidf_transformer.fit_transform(X_tweet_counts)

    # train and test a SVM Classifier
    from sklearn import svm

    return svm.SVC().fit(X_train_tfidf, targets)


def predict(predictor, test_data):
    X_new_counts = count_vect.transform(test_data)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predictedNB = predictor.predict(X_new_tfidf)

    return predictedNB


def test(positive: list, negative: list, seed: int, trainingFunction, use_f):
    """
    Trains and tests our classifiers.
        1. Combine test data
        2. Shuffle test data and verification targets with same seed
        3. Train data with 75% of data and retrieve predictor
        4. Use predictor on other 25% of the data
        5. Retrieve the number of predictions that are correct
        6. Report the percentage correct as well as F-Score
    :param positive: positive tone tweets of the category
    :param negative: negative tone tweets of the category
    :param seed: the random generator seed used on both training and targets
    :param trainingFunction: which type classifier to train
    :param use_f: use the f-score to balance for amounts or have 50/50 pos / negative data
    :return: predictor that we trained
    """
    all_test_data = positive + negative

    # creates a list of target values. Positive entries will be "1" and negative entries will be "0"
    targets = [1] * len(positive)
    targets = targets + ([0] * len(negative))

    random.seed(seed)
    random.shuffle(all_test_data)
    random.seed(seed)
    random.shuffle(targets)

    training_data = all_test_data[:int(.75 * len(all_test_data))]
    test_data = all_test_data[int(.75 * len(all_test_data)):]

    training_targets = targets[:int(.75 * len(targets))]
    test_targets = targets[int(.75 * len(targets)):]

    predictor = None
    if trainingFunction is None:
        predictor = trainNaiveBayes(training_data, training_targets)
    else:
        predictor = trainingFunction(training_data, training_targets)
    predicted = predict(predictor, test_data)

    successes = [1 for prediction, target in zip(predicted, test_targets) if prediction == target]

    besttest = len(successes) / len(test_data)

    print("Best test error accuracy: {:.4f}%".format(besttest))
    print("Best test error f1 score: {:.4f}%".format(f1_score(test_targets, predicted, average='micro')))
    print("Confusion Matrix:")
    print(confusion_matrix(test_targets, predicted))

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
    # print('%r %s' % (doc, category))
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

    net = buildNetwork(trainingds.indim, 10, 10, 10, trainingds.outdim, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(net, dataset=trainingds, learningrate=.65, momentum=.1)

    besttrain = 99.9
    besttest = 99.9
    bestresults = []
    bestclass = []

    for i in range(20):
        trainer.trainEpochs(1)
        trainresult = percentError(trainer.testOnClassData(), trainingds['class'])
        teststuff = trainer.testOnClassData(dataset=testds)
        testresult = percentError(teststuff, testds['class'])
        if testresult < besttest:
            besttest = testresult
            besttrain = trainresult
            bestresults = teststuff
            bestclass = testds['class']

        print("epoch: %2d" % trainer.totalepochs)
        print("train error: %2.2f%%" % trainresult)
        print("test error: %2.2f%%" % testresult)
    print("Best test error accuracy: {:.2f}%".format(besttest))
    print("Best test error f1 score: {:.4f}%".format(f1_score(bestclass, bestresults, average='macro')))
    print("Confusion Matrix:")
    print(confusion_matrix(bestclass, bestresults))

    return besttest


def test_category(name: str, pos_data, neg_data, use_f):
    # IF we don't want to use an F_score to correct for dataset length,
    # this ensures that the length of the two datasets are the same, so that there's a 50% chance of being right by default
    length = max(len(pos_data), len(neg_data)) if use_f else min(len(pos_data), len(neg_data))
    results = []
    print("\n\nCategory: {}".format(name))
    print("-" * len(name))
    # print("{}NN".format(name))
    # results.append(
    #     [testNN([w.text for w in pos_data[:3500]], [w.text for w in neg_data[:3500]], 1)])
    print("{}NB".format(name))
    results.append(
        [test([w.text for w in pos_data[:length]], [w.text for w in neg_data[:length]], 1, trainNaiveBayes, use_f)])
    print("\n{}LR".format(name))
    results.append([
        test([w.text for w in pos_data[:length]], [w.text for w in neg_data[:length]], 1, trainLogisticRegression,
             use_f)])


def dataset_statistics(dataset):
    data = [w.text for w in dataset]
    character_count = 0
    for i in data:
        character_count += len(i)
    avg_length = character_count / len(data)
    print("Average length of document is {} characters".format(avg_length))
    tweet_counts = count_vect.fit_transform(data)
    num_non_zero = tweet_counts.nnz

    print('Dimensions of X_tweet_counts are', tweet_counts.shape)
    print('Number of non-zero elements in x_tweet_counts:', num_non_zero)


def tweet_puller():
    pull_tweets(5000, 'sad')
    pull_tweets(10000, 'courage')
    pull_tweets(10000, 'scared')
    pull_tweets(10000, 'sarcasm')
    pull_tweets(10000, 'serious')
    pull_tweets(10000, 'relaxed')
    pull_tweets(10000, 'stressed')


def test_driver():
    happy = list(read_all_data('happy'))
    sad = list(read_all_data('sad'))
    fearful = list(read_all_data('scary'))
    courageous = list(read_all_data('courage'))
    sarcastic = list(read_all_data('sarcasm'))
    sincere = list(read_all_data('serious'))
    relaxed = list(read_all_data('relaxed'))
    stressed = list(read_all_data('stressed'))

    dataset_statistics(happy)

    print({'happy': len(happy), 'sad': len(sad), 'fearful': len(fearful), 'courageous': len(courageous),
           'sarcastic': len(sarcastic), 'sincere': len(sincere), 'relaxed': len(relaxed), 'stressed': len(stressed)})

    test_category('HappySad', happy, sad, True)
    test_category('Courage', courageous, fearful, True)
    test_category('Sarcasm', sarcastic, sincere, True)
    test_category('Stress', relaxed, stressed, True)


if __name__ == '__main__':
    # tweet_puller()
    test_driver()
