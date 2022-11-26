from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk

train = pd.read_csv('train.csv')
train = train.dropna(subset=['text'])
test  = pd.read_csv('test.csv')

# Save Label
np.save('train_label.npy', train['emotion'])

"""
Bag of Words
"""
from sklearn.feature_extraction.text import CountVectorizer

BOW = CountVectorizer(tokenizer=nltk.word_tokenize) 
# BOW = CountVectorizer(max_features=1024, tokenizer=nltk.word_tokenize) 
BOW.fit(train['text'])
X_train_bow = BOW.transform(train['text'])
X_test_bow = BOW.transform(test['text'])
np.save('train_bow.npy', X_train_bow)
np.save('test_bow.npy', X_test_bow)


"""
TF-IDF
"""
# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_vector = TfidfVectorizer() 
# tfidf_vector.fit(train['text'])
# X_train_tfidf = tfidf_vector.transform(train['text'])
# X_test_tfidf = tfidf_vector.transform(test['text'])
# np.save('train_tfidf_1024.npy', X_train_tfidf)
# np.save('test_tfidf_1024.npy', X_test_tfidf)


"""
Bert
"""
# import torch
# from transformers import AutoModel, AutoTokenizer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")
# model = AutoModel.from_pretrained("vinai/bertweet-large").to(device)
# train_sent = train['text'].tolist()

# batch_size = 256
# X_train_nn = []
# X_test_nn = []
# for i in tqdm(range(0, len(train_sent), batch_size)):
#     sent = train_sent[i:i+batch_size]
#     ids = tokenizer(sent, padding='longest', truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         features = model(ids['input_ids'].to(device), ids['attention_mask'].to(device))
#         X_train_nn.append(features[1].cpu())
        
# test_sent = test['text'].tolist()
# for i in tqdm(range(0, len(test_sent), batch_size)):
#     sent = test_sent[i:i+batch_size]
#     ids = tokenizer(sent, padding='longest', truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         features = model(ids['input_ids'].to(device), ids['attention_mask'].to(device))
#         X_test_nn.append(features[1].cpu())
        
# X_train_nn = torch.cat(X_train_nn)
# X_test_nn  = torch.cat(X_test_nn)
# np.save('train_bert_1024.npy', X_train_nn.numpy())
# np.save('test_bert_1024.npy', X_test_nn.numpy())