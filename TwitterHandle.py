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
    while len(tweet_set) < tweets:
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
        return pickle.load(f)


def read_all_data(tone=None):
    all_data = set()
    for filename in glob.glob("data/*.txt"):
        all_data.update(read_data(filename))

    print(all_data)
    target_data = {tweet for tweet in all_data if tweet.target == tone} if tone else all_data
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


def train(data: list, targets: list):
    count_vect = CountVectorizer()
    count_vect.tokenizer = remove_stop_word_tokenizer
    X_tweet_counts = count_vect.fit_transform(data)

    # Compute term frequencies and store in X_train_tf
    # Compute tfidf feature values and store in X_train_tfidf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_tweet_counts)

    # train and test a Multinomial Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    return MultinomialNB().fit(X_train_tfidf, targets)


def predict(predictor, test_data):
    count_vect = CountVectorizer()
    X_new_counts = count_vect.transform(test_data)
    tfidf_transformer = TfidfTransformer()
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predictedNB = predictor.predict(X_new_tfidf)

    for doc, category in zip(test_data, predictedNB):
         print('%r %s' % (doc, category))
    print('\n')

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
    predicted = predict(predictor, test_data )

    successes = [1 for prediction, target in zip(predicted, test_targets) if prediction == target]
    print(len(successes) / len(test_data))


if __name__ == '__main__':
    # filename = pull_tweets(50000, 'happy')
    # filename = pull_tweets(50000, 'sad')
    positive = read_all_data('happy')
    negative = read_all_data('sad')

    test([w.text for w in positive], [w.text for w in negative], 1)