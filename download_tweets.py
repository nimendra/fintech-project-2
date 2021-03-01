# ####input your credentials here
# consumer_key = 'SCOPBRKeG4nRCEa7XkMoqQ'
# consumer_secret = 'RYInMkLiNyg0iKC3g89Y0f8g8kbNFSsNjZXNYBYILQ'
# access_token = '107275871-PtAa9t7OJX82IEbGjdOzaFMgeCWQyU8haNPhN4mD'
# access_token_secret = 'Q691rTrztjXeYAP5FEA5kZWRhuFRtH8sq6msPgUeD1TzI'

# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = tweepy.API(auth,wait_on_rate_limit=True)
# #####United Airlines
# # Open/Create a file to append data
# csvFile = open('ua.csv', 'a')
# #Use csv Writer
# csvWriter = csv.writer(csvFile)

# all_tweets = []

# for tweet in tweepy.Cursor(api.search,q="#dulux",count=100,
#                            lang="en",
#                            since="2015-11-03").items():
#     print (tweet.created_at, tweet.text)
#     all_tweets.append(tweet.text)
#     csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])