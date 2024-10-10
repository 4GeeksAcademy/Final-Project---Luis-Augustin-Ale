# take off nan, duplicates, 0 lenght tweets, tweets with neutral sentiment (2), convert sentiment value to integer
df=df.dropna()
df=df.drop_duplicates() 
df['tweet_length'] = df['tweet'].apply(lambda x: len(x.split()))
df = df[df['tweet_length'] > 0]
df = df[df['sentiment'] != 2]
df['sentiment'] = df['sentiment'].astype(int)
