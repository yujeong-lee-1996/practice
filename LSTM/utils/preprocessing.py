import torch
from konlpy.tag import Okt
import pickle
import os
import numpy as np


# 전처리 구성 요소
okt = Okt()
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
word_to_index_path = os.path.join(BASE_DIR, '..', 'word_to_index.pkl')
index_to_tag_path = os.path.join(BASE_DIR, '..', 'index_to_tag.pkl')

with open(word_to_index_path, 'rb') as f:
    word_to_index = pickle.load(f)

with open(index_to_tag_path, 'rb') as f:
    index_to_tag = pickle.load(f)

def pad_sequence(tokens, max_len=30):
    features = np.zeros((max_len,), dtype=int)
    if len(tokens) != 0:
        features[:len(tokens)] = np.array(tokens)[:max_len]
    return features

def predict(text, model, device, max_len=30):
    model.eval()
    
    # Tokenize
    tokens = okt.morphs(text)
    tokens = [word for word in tokens if word not in stopwords]

    # Indexing
    indexed = [word_to_index.get(word, word_to_index['<UNK>']) for word in tokens]
    padded = pad_sequence(indexed, max_len)

    # Convert to tensor
    input_tensor = torch.tensor([padded], dtype=torch.long).to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
    predicted_index = torch.argmax(logits, dim=1)
    return index_to_tag[predicted_index.item()]
