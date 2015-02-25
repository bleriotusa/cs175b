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

consumer_key = 'ajGlROl4RpfAg1Kkmm2nxD3Kr'
consumer_secret = 'kstpbSj1NsXtkP7o5as5HxIKIyrHXgJ67T80aggpWtzSNYOYbo'
access_token_key = '175851454-Ih1GVRnJnvwu0NWSTJfon9oT2ehf6sgn73eHYZA5'
access_token_secret = '8560w6tjXiws01Ixf5sPZBVhapp7KbezeDQvi1efTzDtf'


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
tfidf_transformer = TfidfTransformer()

def train(data: list, targets: list):
    X_tweet_counts = count_vect.fit_transform(data)

    # Compute term frequencies and store in X_train_tf
    # Compute tfidf feature values and store in X_train_tfidf
    X_train_tfidf = tfidf_transformer.fit_transform(X_tweet_counts)

    # train and test a Multinomial Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    return MultinomialNB().fit(X_train_tfidf, targets)


random.seed(1)
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

def test(positive: list, negative: list, seed: int):
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

    predictor = train(training_data, training_targets)
    predicted = predict(predictor, test_data)

    for text, prediction, target in zip(test_data, predicted, test_targets):
        print(prediction, target, text)
    successes = [1 for prediction, target in zip(predicted, test_targets) if prediction == target]
    return (len(successes) / len(test_data))


if __name__ == '__main__':
    #
    # pull_tweets(5000, 'courage')
    # pull_tweets(5000, 'scared')
    # pull_tweets(5000, 'sarcasm')
    pull_tweets(5000, 'serious')
    # pull_tweets(5000, 'relaxed')
    # pull_tweets(5000, 'stressed')

    happy = list(read_all_data('happy'))
    sad = list(read_all_data('sad'))
    fearful = list(read_all_data('scary'))
    courageous = list(read_all_data('courage'))
    sarcastic = list(read_all_data('sarcasm'))
    sincere = list(read_all_data('serious'))
    relaxed = list(read_all_data('relaxed'))
    stressed = list(read_all_data('stressed'))


    # Ensures that the length of the two datasets are the same, so that there's a 50% chance of being right by default
    happylen = min(len(happy), len(sad))
    couragelen = min(len(fearful), len(courageous))
    sarcasmlen = min(len(sarcastic), len(sincere))
    relaxedlen = min(len(relaxed), len(stressed))


    print(len(happy), len(sad), len(fearful), len(courageous),
          len(sarcastic), len(sincere), len(relaxed), len(stressed))

    results = [test([w.text for w in happy[:happylen]], [w.text for w in sad[:happylen]], 1)]
    results.append(test([w.text for w in courageous[:couragelen]], [w.text for w in fearful[:couragelen]], 1))
    results.append(test([w.text for w in sarcastic[:sarcasmlen]], [w.text for w in sincere[:sarcasmlen]], 1))
    results.append(test([w.text for w in stressed[:relaxedlen]], [w.text for w in relaxed[:relaxedlen]], 1))
    print(results)