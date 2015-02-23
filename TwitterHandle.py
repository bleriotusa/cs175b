__author__ = 'Michael'
import ast
import time
import glob
import nltk
from nltk.corpus import stopwords
stopwordslist = stopwords.words('english')
from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterRestPager
from datetime import datetime
import pickle

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

    count = 0
    tweet_list = []

    for item in r.get_iterator():
        tweet = Tweet()
        if count >= tweets:
            break
        if 'text' in item:
            tweet.hashtags = [hashtag['text'] for hashtag in item['entities']['hashtags']]
            tweet.text = item['text'].replace('\n', ' ')
            tweet.target = hashtag

            tweet_list.append(tweet)
            print(tweet.hashtags, tweet.text, tweet.target)
            count += 1
        elif 'message' in item and item['code'] == 88:
            print('SUSPEND, RATE LIMIT EXCEEDED: %s\n' % item['message'])
            time.sleep(16 * 60)

    pickle.dump(tweet_list, data_file, 2)
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


def read_all_data():
    all_data = []
    for filename in glob.glob("data/*.txt"):
        all_data.extend(read_data(filename))

    for tweet in all_data:
        print('{}: {}\n\t{}'.format(tweet.target, tweet.hashtags, tweet.text))

    return all_data

# Remove stop words
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
default_tokenizer_function = count_vect.build_tokenizer()

# Override default countvectorizer tokenizer to remove stopwords
def remove_stop_word_tokenizer(s):
    words = default_tokenizer_function(s)
    words = list(w for w in words if w.lower() not in stopwordslist)
    return words


if __name__ == '__main__':
    # filename = pull_tweets(50, 'happy')
    # print(read_data('data/'+str(filename)+'.txt'))
    data = read_all_data()
    print(len(data))