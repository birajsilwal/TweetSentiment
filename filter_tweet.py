import re


'''
this class filters (removes tweet tags, punctuations,
irrelevant words such as a, an, the) from the tweets
and returns array of clean filtered tweets
'''
class FilterTweets:
    # opening the testing file
    tweet_testing = open('Tweets/Tweets_election_trial.txt')
    final_tweets = []


    def filterTweets(self):
        tweets = []
        for tweet in self.tweet_testing:
            tweet = tweet.lower()
            tweet = re.sub(":|;|\(|\)|!|\'s|\"|\.|n\'t|,|\?|i\'m|i\'ve", "", tweet)
            tweet = re.sub(" an | the | is | of | a | i | was | and |"
                           " on | in | off | all | it | me | you | to |"
                           " into | we | your | that | they | can | could |"
                           " should | do | does | for | my | at | so | So |"
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
                           " should | do | does | for | my | at | so | So |"
                           " if | has | have | had | from | such | are |"
                           " not | this | now | but | go | day |"
                           " - | up | down | these | today | lol |"
                           " lmao | af | get | got | here | there | who |"
                           " what | am | no | why | with | us | our | bro ", " ", tweet)
            tweets1.append(tweet)
            print(tweet)


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
            self.final_tweets.append(tweet)
            # print(tweet)
        return self.final_tweets

