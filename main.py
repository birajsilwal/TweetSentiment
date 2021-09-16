"""
Biraj Silwal
101797855
"""
import math
import re
from typing import List, Dict
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


# sentiment enum
class Sentiment:
    NEGATIVE = "NEGATIVE"
    POSITIVE = "POSITIVE"


'''
this class make an object of tweet
tweet.tweet gives text
tweet.zero_or_one gives 0 or 1 (negative or positive sentiment)
'''


class Tweet:

    def __init__(self, tweet, zero_or_one, username, user_screen_name, time, is_retweet):
        self.tweet = tweet
        self.zero_or_one = zero_or_one
        self.username = username
        self.user_screen_name = user_screen_name
        self.time = time
        self.is_retweet = is_retweet
        self.sentiment = self.get_sentiment()

    # this method returns sentiment
    def get_sentiment(self):
        if self.zero_or_one == '0':
            return "NEGATIVE"
        elif self.zero_or_one == '1':
            return "POSITIVE"

    def tweet_obj(self, filtered_tweet):
        tweets = []
        for tweet in filtered_tweet:
            string_arr = tweet.split()
            zero_or_one = string_arr.pop(0)

            tweet1 = ' '
            tweet1 = tweet1.join(string_arr)
            tweets.append(Tweet(tweet1, zero_or_one, None, None, None, None))
        return tweets


'''
A Bayes classifier
2 implementations
- 1. using builtin sklearn (Gaussian Naive Bayes)
- 2. using custom model
'''


class BayesClassifier:
    """
    Attributes:
        tweets_training - contains all the tweets from training dataset
        tweets_testing - contains all the tweets from testing dataset

        train_x - input tweet for training
        train_y - output sentiment for training
        test_x - input tweet for testing
        test_y - output sentiment for testing

        train_x_raw - input raw tweet for training
        train_y_raw - output sentiment for training
        test_x_raw - input raw tweet for testing
        test_y_raw - output sentiment for testing

        train_x_vectors - input vectors of train data
        test_x_vectors - input vectors of test data
        vectorizer - Count vectorizer for Bag of words

        pos_dict - dictionary of positive tweets (key is feature, value is it's occurrence)
        neg_dict - dictionary of negative tweets (key is feature, value is it's occurrence)
    """

    def __init__(self):
        # initialization of variables
        self.tweets_training = []
        self.tweets_testing = []
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.train_x_raw = []
        self.train_y_raw = []
        self.test_x_raw = []
        self.test_y_raw = []
        self.train_x_vectors = []
        self.test_x_vectors = []
        self.vectorizer = CountVectorizer()
        self.pos_dict = {}
        self.neg_dict = {}

    """
    this method filters (removes tweet tags, punctuations,
    irrelevant words such as a, an, the) from the tweets
    and returns array of clean filtered tweets
    """

    def filter_tweets(self, file_path):
        final_tweets = []
        tweets = []
        tweets_raw = open(file_path)
        for tweet in tweets_raw:
            tweet = tweet.lower()
            tweet = re.sub(":|;|\(|\)|!|\'s|\"|\.|n\'t|,|\?|i\'m|i\'ve", "", tweet)
            tweet = re.sub(" an | the | is | of | a | i | was | and |"
                           " on | in | off | all | it | me | you | to |"
                           " into | we | your | that | they | can | could |"
                           " should | do | does | for | my | at | so |"
                           " if | has | have | had | from | such | are |"
                           " not | this | now | but | go | day |"
                           "-|_| up | down | these | today | lol |"
                           " lmao | af | get | got | here | there | who |"
                           " what | am | no | why | with | us | our | bro |"
                           " too | then | ur | zero | ah | see | saw ", " ", tweet)
            tweets.append(tweet)

        # cleaning up twice because of space issue
        tweets1 = []
        for tweet in tweets:
            tweet = tweet.lower()
            tweet = re.sub(" an | the | is | of | a | i | was | and |"
                           " on | in | off | all | it | me | you | to |"
                           " into | we | your | that | they | can | could |"
                           " should | do | does | for | my | at | so |"
                           " if | has | have | had | from | such | are |"
                           " not | this | now | but | go | day |"
                           "-|_| up | down | these | today | lol |"
                           " lmao | af | get | got | here | there | who |"
                           " what | am | no | why | with | us | our | bro |"
                           " too | then | ur | zero | ah | see | saw ", " ", tweet)
            tweets1.append(tweet)

        '''
        if there contains a substring starting with http or @,
        remove that substring from the tweet
        '''
        for str in tweets1:
            index = 0
            string_arr = str.split()
            for string in string_arr:
                if 'http' in string or '@' in string:
                    string_arr[index] = string.replace(string, '')
                index += 1

            tweet = ' '
            tweet = tweet.join(string_arr)
            final_tweets.append(tweet)
        return final_tweets

    # filters and creates tweet object
    def process_tweets(self):
        tweet_obj = Tweet(None, None, None, None, None, None)

        # filter the tweets
        training_tweet_filepath = 'Tweets/noCR_train.txt'
        tweets_train_raw = open(training_tweet_filepath)
        tweets_train_filtered = self.filter_tweets(training_tweet_filepath)

        testing_tweet_filepath = 'Tweets/noCR_test.txt'
        tweets_test_raw = open(testing_tweet_filepath)
        tweets_test_filtered = self.filter_tweets(testing_tweet_filepath)

        # tweet training and testing datasets (raw - unfiltered)
        tweets_training_raw = tweet_obj.tweet_obj(tweets_train_raw)
        tweets_testing_raw = tweet_obj.tweet_obj(tweets_test_raw)

        # tweet training and testing datasets
        self.tweets_training = tweet_obj.tweet_obj(tweets_train_filtered)
        self.tweets_testing = tweet_obj.tweet_obj(tweets_test_filtered)

        # x and y values for both training and testing dataset for raw tweets
        self.train_x_raw = [x.tweet for x in tweets_training_raw]
        self.train_y_raw = [x.sentiment for x in tweets_training_raw]

        self.test_x_raw = [x.tweet for x in tweets_testing_raw]
        self.test_y_raw = [x.sentiment for x in tweets_testing_raw]

        # x and y values for both training and testing dataset
        self.train_x = [x.tweet for x in self.tweets_training]
        self.train_y = [x.sentiment for x in self.tweets_training]

        self.test_x = [x.tweet for x in self.tweets_testing]
        self.test_y = [x.sentiment for x in self.tweets_testing]

    # creates a dictionary of tweet with
    # key is the word in the tweet
    # value is the frequency (occurrence) of the word (key)
    def update_dict(self):
        self.pos_dict = {}
        self.neg_dict = {}
        dict_tweet = {}
        val = 1

        for tweet in self.tweets_training:
            if tweet.sentiment == Sentiment.POSITIVE:
                tweet = tweet.tweet.split()
                for word in tweet:
                    key = word
                    if key in self.pos_dict:
                        update_val = self.pos_dict[key] + 1
                        update_dict = {key: update_val}
                        self.pos_dict.update(update_dict)
                    else:
                        self.pos_dict[key] = val
            elif tweet.sentiment == Sentiment.NEGATIVE:
                tweet = tweet.tweet.split()
                for word in tweet:
                    key = word
                    if key in self.neg_dict:
                        update_val = self.neg_dict[key] + 1
                        update_dict = {key: update_val}
                        self.neg_dict.update(update_dict)
                    else:
                        self.neg_dict[key] = val

    # this method creates bag of words vectors
    def vectorize(self) -> None:

        # for raw tweet
        self.train_x_raw_vectors = self.vectorizer.fit_transform(self.train_x_raw)
        self.test_x_raw_vectors = self.vectorizer.transform(self.test_x_raw)

        # for filtered tweet
        self.train_x_vectors = self.vectorizer.fit_transform(self.train_x)
        self.test_x_vectors = self.vectorizer.transform(self.test_x)

    # takes list of tweets and classifies them as positive or negative
    def classify(self, tweets):
        """Classifies given text as positive or negative"""
        # sum of sum of all the frequencies of the feature
        # in the positive and negative dictionary
        # this is used to calculate probability later on
        total_pos_freq = 0
        total_neg_freq = 0
        for key, val in self.pos_dict.items():
            total_pos_freq += val
        for key, val in self.neg_dict.items():
            total_neg_freq += val

        result = []

        # for each word in the tweet, calculate the probability of it
        # being positive and negative tweet
        # here 1 is added to avoid probability being 0
        for tweet in tweets:
            # this variable stores the total
            # positive and negative probability.
            pos_prob = 0
            neg_prob = 0
            tweet = tweet.split()
            for word in tweet:
                if word in self.pos_dict:
                    pos_prob += math.log((self.pos_dict[word] + 1) / total_pos_freq)
                if word in self.neg_dict:
                    neg_prob += math.log((self.neg_dict[word] + 1) / total_neg_freq)

            if pos_prob > neg_prob:
                result.append(Sentiment.POSITIVE)
            else:
                result.append(Sentiment.NEGATIVE)

        return result

    # gaussian naive bayes classification using sklearn
    def classify_naive_bayes(self, test_set):
        clf = GaussianNB()

        clf.fit(self.train_x_raw_vectors.toarray(), self.train_y_raw)
        # printing accuracy for the training and testing dataset
        print("Before filter")
        print('Accuracy of train:', clf.score(self.train_x_raw_vectors.toarray(), self.train_y_raw))
        print('Accuracy of test:', clf.score(self.test_x_raw_vectors.toarray(), self.test_y_raw))
        # calculating F1 score
        print('F1 score: ', f1_score(self.test_y_raw, clf.predict(self.test_x_raw_vectors.toarray()), average=None,
                                     labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
        print('')

        clf.fit(self.train_x_vectors.toarray(), self.train_y)
        # printing accuracy for the training and testing dataset
        print("After filter")
        print('Accuracy of train:', clf.score(self.train_x_vectors.toarray(), self.train_y))
        print('Accuracy of test:', clf.score(self.test_x_vectors.toarray(), self.test_y))
        # calculating F1 score
        print('F1 score: ', f1_score(self.test_y, clf.predict(self.test_x_vectors.toarray()), average=None,
                                     labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))
        print('')

        # test your custom tweet
        new_test = self.vectorizer.transform(test_set)
        # print('Testing array:', test_set)
        sentiment_trial = clf.predict(new_test.toarray())

        print('Sentiment for testing array:', clf.predict(new_test.toarray()))

        return sentiment_trial


    # takes observed output and classified output as a parameter and returns the accuracy
    def accuracy(self, observed_output, classified_output):
        TP = 0  # Observation is positive, and is predicted to be positive.
        FN = 0  # Observation is positive, but is predicted negative.
        TN = 0  # Observation is negative, and is predicted to be negative.
        FP = 0  # Observation is negative, but is predicted positive.

        for index, line in enumerate(classified_output):
            if observed_output[index] == Sentiment.POSITIVE:
                if observed_output[index] == Sentiment.POSITIVE and classified_output[index] == Sentiment.POSITIVE:
                    TP += 1
                else:
                    FN += 1
            elif observed_output[index] == Sentiment.NEGATIVE:
                if observed_output[index] == Sentiment.NEGATIVE and classified_output[index] == Sentiment.NEGATIVE:
                    TN += 1
                else:
                    FP += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        return accuracy



# this class is related to tweets around election time
class Election:
    tweet_election = []
    tweet_election_tweet_only = []
    tweets = open('Tweets/Tweets_election_trial.txt')
    for tweet in tweets:
        tweet = tweet.split('\t')
        tweet_election.append(Tweet(tweet[4], None, tweet[0], tweet[1], tweet[2], tweet[3]))

    for tweet in tweet_election:
        tweet_election_tweet_only.append(tweet.tweet)


if __name__ == "__main__":
    bc = BayesClassifier()
    bc.process_tweets()
    bc.update_dict()
    bc.vectorize()

    pos_counter_train = 0
    neg_counter_train = 0

    # ## sentiment of training dataset classified by custom naive bayes classifier
    train_sentiment = bc.classify(bc.train_x)
    for item in train_sentiment:
        if item == Sentiment.POSITIVE:
            pos_counter_train += 1
        else:
            neg_counter_train += 1

    pos_counter_test = 0
    neg_counter_test = 0

    # ## sentiment of training dataset classified by custom naive bayes classifier
    test_sentiment = bc.classify(bc.test_x)
    for item in test_sentiment:
        if item == Sentiment.POSITIVE:
            pos_counter_test += 1
        else:
            neg_counter_test += 1

    # number of positive and negative outcomes
    # print('Training data -> ', 'pos:', pos_counter_train, 'neg:', neg_counter_train)
    # print('Training data -> ', 'pos:', pos_counter_test, 'neg:', neg_counter_test)

    # ## enter your testing tweet here
    input_test_set = ['Thanks for your contribution and kind comment',
                      'waste of time',
                      'Sending some laughter',
                      'I hate you.',
                      'I love this world']

    # ## classification with custom built naive bayes
    result = bc.classify(input_test_set)
    # result = bc.classify_naive_bayes(input_test_set)
    print(result)

    # ******************
    # Accuracy
    # ******************

    train_accuracy = bc.accuracy(bc.train_y, train_sentiment)
    test_accuracy = bc.accuracy(bc.test_y, test_sentiment)
    print("Train accuracy: ", train_accuracy)
    print("Test accuracy: ", test_accuracy)




    # ## processing election tweets
    positive_time = []
    negative_time = []

    election = Election()
    tweet_election_tweet_only = election.tweet_election_tweet_only
    tweet_election = election.tweet_election



    # ****************
    # UNCOMMENT everything below here if you want to run election tweets
    # ****************

#     # ## the output is by using sklearn library
#     # sentiment_output_sklern = bc.classify_naive_bayes(tweet_election_tweet_only)
#
#     # ## the output is by using custom naive bayes model
#     sentiment_output = bc.classify(tweet_election_tweet_only)
#
#     for index, line in enumerate(sentiment_output):
#         if line == 'POSITIVE':
#             positive_time.append(tweet_election[index].time.split(' ')[0])
#         else:
#             negative_time.append(tweet_election[index].time.split(' ')[0])
#
#     print('Number of positive sentiments during election: ', len(positive_time))
#     print('Number of negative sentiments during election: ', len(negative_time))
#
#     # ##sorting time
#     positive_time.sort()
#     negative_time.sort()
#
#     '''creating dictionary for positive sentiment time
#     and negative sentiment time for histogram '''
#     dict_pos_time = {}
#     dict_neg_time = {}
#
#     val = 1
#     for line in positive_time:
#         key = line
#
#         if key in dict_pos_time:
#             update_val = dict_pos_time[key] + 1
#             update_dict = {key: update_val}
#             dict_pos_time.update(update_dict)
#         else:
#             dict_pos_time[key] = val
#
#     for line in negative_time:
#         key = line
#
#         if key in dict_neg_time:
#             update_val = dict_neg_time[key] + 1
#             update_dict = {key: update_val}
#             dict_neg_time.update(update_dict)
#         else:
#             dict_neg_time[key] = val
#
# # ## time of positive sentiment
# plt.bar(list(dict_pos_time.keys()), dict_pos_time.values(), color='g')
# plt.xticks(rotation=90)
# plt.gcf().subplots_adjust(bottom=0.25)
# plt.legend("Positive")
# plt.show()
#
# # ## time of negative sentiment
# plt.bar(list(dict_neg_time.keys()), dict_neg_time.values(), color='r')
# plt.xticks(rotation=90)
# plt.gcf().subplots_adjust(bottom=0.25)
# plt.legend("Negative")
# plt.show()
