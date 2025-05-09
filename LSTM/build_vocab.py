# build_vocab.py

import re
from collections import Counter
import pickle
import torch

# 텍스트 전처리 + 토큰화 함수
def tokenizer(text):
    text = re.sub(r"[^가-힣0-9\s]", "", text)
    return text.strip().split()

# vocab 생성 함수
def build_vocab(file_path, vocab_size=10000):
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # 첫 줄 skip
        for line in f:
            try:
                _, text, _ = line.strip().split('\t')
                tokens = tokenizer(text)
                counter.update(tokens)
            except:
                continue

    most_common = counter.most_common(vocab_size - 2)
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i

    return vocab

# 🔧 vocab 생성 및 저장
if __name__ == "__main__":
    vocab = build_vocab("data-files/ratings_train.txt")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("✅ vocab.pkl 생성 완료!")


# 📦 Flask에서 사용할 전처리 함수 
def preprocess_input(text, vocab, max_len=100):
    tokens = tokenizer(text)
    ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab['<pad>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor([ids], dtype=torch.long)

