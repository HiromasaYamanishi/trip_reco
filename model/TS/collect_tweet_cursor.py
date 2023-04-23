import tweepy
 
import tweepy
from datetime import datetime, timezone
import pandas as pd
import time
from tqdm import tqdm
consumer_key = 'l8V5FA48CTUmU2zGXJjU548BU'
consumer_secret = 'royCed2iqk5yS0pvukBoCBdAYkOBDvo8U6LuFt4yrqTWeqVlfL'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAIUwiQEAAAAAlMyR6aYvSVOGTkS%2BeL%2FKcqBjz28%3DGbYSx4Bd7EC1xIsKdPoLzVamILxQ1fVb16Ht7CzGE1dlMa2RPl'
access_token = '1577902360009277441-SI6jU7MCHq1Ho4XbmocJlMRbbtvDo3'
access_token_secret = 'hojQOMb2UowPidCKByIpDT6RPAncritKmYfDE8q8wmXyR'
df = pd.read_csv('./data/tokyo_mesh.csv')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def search_tweet_by_day(month, day, lat_center, lon_center, lat_min, lat_max, lon_min, lon_max, cityname, meshid):
    q='(あ OR い OR う OR え OR お OR か OR き OR く OR け OR こ OR さ OR し OR す OR せ OR そ OR た OR ち OR つ OR て OR と OR な OR に OR ぬ OR ね OR の OR は OR ひ OR ふ OR へ OR ほ OR ま OR み OR む OR め OR も OR や OR ゆ OR よ OR わ OR を OR ん OR , OR 。)'
    until = f'2022-{month}-{day}_00:00:00_JST'
    #until = datetime(2022, 11,28,15,0,0,0,timezone.utc)
    limit = datetime(2022, month,day-2,15,0,0,0,timezone.utc)
    meshids, ids, texts, lat_centers, lon_centers, created_ats, citynames = [],[],[],[],[],[],[]
    for i,tweet in enumerate(tweepy.Cursor(api.search_tweets, count=100, q=q, geocode=f"{lat_center},{lon_center},0.8km", until=until).items(2000)):
        #tweets=api.search_tweets(count=100, q=q, geocode="35.736054,139.7122,1km", until=until)
        #print(f'No {i}')
        #print(tweet.text)
        #print(tweet.created_at)
        if tweet.place is not None:
            bounding_box = tweet.place.bounding_box.coordinates
            #print(bounding_box)
            lon_min_tmp = bounding_box[0][0][0]
            lon_max_tmp = bounding_box[0][1][0]
            lat_min_tmp = bounding_box[0][0][1]
            lat_max_tmp = bounding_box[0][2][1]
            lon_center_tmp = (lon_max_tmp+lon_min_tmp)/2
            lat_center_tmp = (lat_max_tmp+lat_min_tmp)/2
            print(lon_min_tmp, lon_max_tmp, lat_min_tmp, lat_max_tmp)
            print(lat_min, lat_center_tmp, lat_max, lon_min, lon_center_tmp, lon_max)
            if (lat_min < lat_center_tmp < lat_max) and (lon_min < lon_center_tmp < lon_max):
                ids.append(tweet.id)
                texts.append(tweet.text)
                lat_centers.append(lat_center_tmp)
                lon_centers.append(lon_center_tmp)
                created_ats.append(tweet.created_at.strftime('%Y-%m-%d_%H:%M:%S'))
                citynames.append(cityname)
                meshids.append(meshid)
        if tweet.created_at < limit:
            break
    df = pd.DataFrame({'meshid': meshids,'id':ids, 'text':texts, 'latitude':lat_centers, 'longitude':lon_centers, 'created_at':created_ats, 'cityname':citynames})
    df.to_csv('data/tweet/geotweet.csv', mode='a', header=False)
        #print(lon_center, lat_center)
        #print(tweet.id)
        #until = tweet.created_at.strftime('%Y-%m-%d_%H:%M:%S')
        #print(until)
        #print('-'*100)

if __name__=='__main__':
    df_mesh = pd.read_csv('./data/tokyo_mesh.csv')
    for index in tqdm(range(18,len(df_mesh))):
        for day in range(22, 31):
            print(index, day)
            search_tweet_by_day(11,day, df_mesh.loc[index, 'lat_center'], df_mesh.loc[index, 'lon_center'], df_mesh.loc[index, 'lat_min'], df_mesh.loc[index, 'lat_max'], df_mesh.loc[index, 'lon_min'], df_mesh.loc[index, 'lon_max'], df_mesh.loc[index, 'cityname'], df_mesh.loc[index, 'mesh1kmid'])
            time.sleep(10)