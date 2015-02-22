__author__ = 'Michael'

import time
from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterRestPager
from datetime import datetime
import ast

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
            data = '{}\n{}\n'.format([hashtag['text'] for hashtag in item['entities']['hashtags']],
                                     item['text'].replace('\n', ' '))
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

    with open(filename) as f:
        tweets = []
        tweet = Tweet()
        for line in f:
            if line.startswith("['"):
                tweet = Tweet()
                # print(line)
                tweet.hashtags = ast.literal_eval(line.strip())  # method to evaluate list strings into python lists
            else:
                tweet.text = line.strip()
                tweets.append(tweet)

        for tweet in tweets:
            print('{}\n\t{}'.format(tweet.hashtags, tweet.text))


if __name__ == '__main__':
    filename = pull_tweets(18000, 'happy')
    read_data('data/{}.txt'.format(filename))