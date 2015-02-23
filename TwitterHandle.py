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

    data_file = open('data/{}{}'.format(str(start_time), '.txt'), 'w+')
    r = TwitterRestPager(api, 'search/tweets', {'q': '#{}'.format(hashtag), 'count': 100, 'lang': 'en'})

    count = 0
    for item in r.get_iterator():
        if count >= tweets:
            break
        if 'text' in item:
            data = '{}\n{}\n{}\n'.format([hashtag['text'] for hashtag in item['entities']['hashtags']],
                                     item['text'].replace('\n', ' '), hashtag)
            print(count, data)
            data_file.write(data)
            count += 1
        elif 'message' in item and item['code'] == 88:
            print('SUSPEND, RATE LIMIT EXCEEDED: %s\n' % item['message'])
            time.sleep(16 * 60)

    data_file.close()
    print(datetime.now() - start_time)
    return start_time


def read_data(filename: str) -> list:
    """
    :param filename: file to read from
    :return: list of Tweet objects
    """
    num_lines_data = 3
    with open(filename) as f:
        tweets = []
        tweet = Tweet()

        for counter, line in enumerate(f):
            if counter % num_lines_data is 0:
                tweet = Tweet()
                # print(line)
                tweet.hashtags = ast.literal_eval(line.strip())  # method to evaluate list strings into python lists
            elif counter % num_lines_data is 1:
                tweet.text = line.strip()
            elif counter % num_lines_data is 2:
                tweet.target = line.strip()
                tweets.append(tweet)

        for tweet in tweets:
            print('{}\n\t{}'.format(tweet.hashtags, tweet.text))

def read_all_data():
    for filename in glob.glob("data/*.txt"):
        read_data(filename)




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
    filename = pull_tweets(180, 'happy')
    read_all_data()