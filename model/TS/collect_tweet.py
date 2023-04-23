import tweepy
consumer_key = 'l8V5FA48CTUmU2zGXJjU548BU'
consumer_secret = 'royCed2iqk5yS0pvukBoCBdAYkOBDvo8U6LuFt4yrqTWeqVlfL'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAIUwiQEAAAAAlMyR6aYvSVOGTkS%2BeL%2FKcqBjz28%3DGbYSx4Bd7EC1xIsKdPoLzVamILxQ1fVb16Ht7CzGE1dlMa2RPl'
access_token = '1577902360009277441-SI6jU7MCHq1Ho4XbmocJlMRbbtvDo3'
access_token_secret = 'hojQOMb2UowPidCKByIpDT6RPAncritKmYfDE8q8wmXyR'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
h = 'あいうえお'\
    'かきくけこ'\
    'さしすせそ'\
    'たちつてと'\
    'なにぬなの'\
    'はひふへほ'\
    'まみむめも'\
    'やゆよ'\
    'らりるれろ'\
    'わをん'
q = '('
for h_ in h:
    q+=h_
    q+=' OR '
q=q[:-1]
q+=')'
print('query',q)
q='(あ OR い OR う OR え OR お OR か OR き OR く OR け OR こ OR さ OR し OR す OR せ OR そ OR た OR ち OR つ OR て OR と OR な OR に OR ぬ OR ね OR の OR は OR ひ OR ふ OR へ OR ほ OR ま OR み OR む OR め OR も OR や OR ゆ OR よ OR わ OR を OR ん OR , OR 。)'
tweets=api.search_tweets(count=20, q=q, geocode="35.736054,139.7122,1km", until='2022-11-29_00:00:00')
for tweet in tweets:
    #print(tweet)
    print(tweet.text)
    print(tweet.created_at)
    if tweet.place is not None:
        print(tweet.place.bounding_box.coordinates)
    print(tweet.id)
    print('-'*100)