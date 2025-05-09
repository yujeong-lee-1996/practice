import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 모델은 숫자만 이해하기 때문에 각 단어 정수를 의미를 담은 고차원 벡터로 변환 

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # LSTM Layer : 순차 정보 이해 

        self.fc = nn.Linear(hidden_dim, output_dim)
        # 마지막 hidden state를 받아서 클래스(긍정/부정) 확률로 변환 

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        # 정수 시퀀스를 단어 벡터 시퀀스로 변환 

        # LSTM은 (hidden state, cell state)의 튜플을 반환합니다
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: (batch_size, seq_length, hidden_dim), hidden: (1, batch_size, hidden_dim)
        #  LSTM 에 넣고 순차 정보 반영한 hidden state 얻기 

        last_hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        # 마지막 timestep 의 hidden vector 만 추출 (문장의미 요약 )

        logits = self.fc(last_hidden)  # (batch_size, output_dim)
        # → 긍정/부정 확률 score 계산 
        
        return logits