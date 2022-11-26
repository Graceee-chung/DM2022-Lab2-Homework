import os
import re
import emoji
import pandas as pd
import itertools
from emoji import demojize
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup
"""
Preprocess Dataset
"""

def load_smileys():
    return {
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }

def load_contractions():
    
    return {
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "can't":"cannot",
        "'cause":"because",
        "couldn't":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "didn't":"did not",
        "doesn't":"does not",
        "don't":"do not",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "hasn't":"has not",
        "haven't":"have not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "i'd":"i would",
        "i'll":"i will",
        "i'm":"i am",
        "i'm'a":"i am about to",
        "i'm'o":"i am going to",
        "isn't":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "i've":"i have",
        "kinda":"kind of",
        "let's":"let us",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wanna":"want to",
        "wasn't":"was not",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "y'all":"you all",
        "you'd":"you would",
        "you'll":"you will",
        "you're":"you are",
        "you've":"you have",
        "whatcha":"what are you",
        "luv":"love",
        "sux":"sucks"
        }

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = TOKENIZER.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

def remove_accents(text):
    if 'ø' in text or  'Ø' in text: return text   
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


CONTRACTIONS = load_contractions()
SMILEY = load_smileys()
TOKENIZER = TweetTokenizer()

def clean_text(text):
    text = text.lower()
    text = text.replace("’","'")
    words = text.split()
    text = " ".join([CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words])
    
    text = normalizeTweet(text)
    
    text = BeautifulSoup(text).get_text()
    text = text.replace('\x92',"'")
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    words = text.split()
    text = " ".join([SMILEY[word] if word in SMILEY else word for word in words])
    
    text = normalizeTweet(text)
    
    text = re.sub(r'http\S+', ' ', text) 
    text = re.sub(r'@\w+',' ',text)      
    text = re.sub(r'#', ' ', text)       
    text = re.sub(r'<.*?>',' ', text)   
    text = re.sub(r'[\.\,\!\?\:\;\-\=\_\~]',' ', text) 
    text = re.sub(r'<LH>\s', r'', text)
    text = re.sub(r'([^a-z])\1+', r'\1', text)
    
    text = remove_accents(text)
    text = ' '.join(text.split())
    text = text.replace('"', ' ')
    
    text = emoji.demojize(text, delimiters=("", ""))
    return text


# Read Dataset 
train_test = pd.read_csv('data_identification.csv').drop_duplicates()
emotion = pd.read_csv('emotion.csv').drop_duplicates()
raw_data = pd.read_json('tweets_DM.json', lines=True)

# Split Dataset
train = train_test[train_test['identification'] == 'train']
test = train_test[train_test['identification'] == 'test']

# Sort Dataset
train = train.sort_values(by=['tweet_id'])
test = test.sort_values(by=['tweet_id'])
emotion = emotion.sort_values(by=['tweet_id'])
print(len(train), len(emotion))

# Get Text
source = list(raw_data['_source'].values)
source = [i['tweet'] for i in source]
source = pd.DataFrame(data=source ,columns=['tweet_id', 'hashtags', 'text'])
source = source.sort_values(by=['tweet_id'])
source['text'] = source.text.apply(lambda x: clean_text(x))

train = train.merge(source, on='tweet_id', how='left')
train = train.merge(emotion, on='tweet_id', how='left')
test  = test.merge(source, on='tweet_id', how='left')

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)