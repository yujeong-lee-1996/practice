# # views/word2vec_view.py
from flask import Blueprint, request, render_template
from gensim.models import Word2Vec
import os
from gensim.models import Word2Vec

import os
from gensim.models import Word2Vec


# 현재 파일 위치 기준 → W2V 폴더로 올라감
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # W2V/views
BASE_DIR = os.path.dirname(BASE_DIR)  # W2V

# 올바른 경로로 수정
model_path = os.path.join(BASE_DIR, "model", "my_word2vec.model")
model = Word2Vec.load(model_path)

word2vec_bp = Blueprint('word2vec', __name__, url_prefix="/word2vec")
# model = Word2Vec.load("model/my_word2vec.model")
main_bp = Blueprint('main', __name__, url_prefix="/")

# @word2vec_bp.route("/", methods=["GET", "POST"])
# def w2v():
#     result = []
#     if request.method == "POST":
#         word = request.form.get("word", "")
#         try:
#             result = model.wv.most_similar(word, topn=10)
#         except KeyError:
#             result = [("❌ 단어 없음", 0.0)]
#     return render_template("w2v.html", result=result)

@word2vec_bp.route("/", methods=["GET", "POST"])
def w2v():
    result = []
    if request.method == "POST":
        pos1 = request.form.get("positive1", "").strip()
        pos2 = request.form.get("positive2", "").strip()
        neg = request.form.get("negative", "").strip()

        try:
            result = model.wv.most_similar(
                positive=[pos1, pos2], 
                negative=[neg],
                topn=10
            )
        except KeyError:
            result = [("❌ 해당 단어가 벡터에 없음", 0.0)]
    return render_template("w2v.html", result=result)