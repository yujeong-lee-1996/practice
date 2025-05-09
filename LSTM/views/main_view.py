import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, render_template, request
import torch
from models.lstm_model import TextClassifier
from utils.preprocessing import predict  

main_bp = Blueprint('main', __name__, url_prefix="/")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 경로 설정 및 로드
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_checkpoint.pth")

model = TextClassifier(vocab_size=28200, embedding_dim=100, hidden_dim=128, output_dim=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 라우트 정의
@main_bp.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_text = request.form['text_input']
        prediction = predict(input_text, model, device)  

    return render_template('index.html', prediction=prediction)