import tweepy
from transformers import pipeline

# Test commit
#-----------------FETCHER EXAMPLE-----------------#


BEARER_TOKEN = ''
def get_tweet_content(tweet_url):
    tweet_id = tweet_url.split('/')[-1]

    client = tweepy.Client(bearer_token=BEARER_TOKEN)

    tweet = client.get_tweet(tweet_id, tweet_fields=['text'])
    
    if not tweet.data:
        raise Exception("Could not fetch the tweet. Ensure the tweet ID is correct and the tweet is public.")
    
    tweet_text = tweet.data['text']
    
    return tweet_text

def analyze_sentiment(text):
    sentiment_analysis = pipeline("sentiment-analysis")
    result = sentiment_analysis(text)[0]
    return result

def main(tweet_url):
    tweet_text = get_tweet_content(tweet_url)
    print(f"Tweet: {tweet_text}\n")
    
    sentiment = analyze_sentiment(tweet_text)
    print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']:.4f}")

if __name__ == "__main__":
    tweet_url = "https://twitter.com/elonmusk/status/1790124424986849471"
    main(tweet_url)