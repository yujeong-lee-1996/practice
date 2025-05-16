from views.main_view import main_bp
from views.lstm_view import lstm_bp
from views.word2vec_view import word2vec_bp
from views.chat_view import chat_bp

from flask import Flask

def create_app():
    app = Flask(__name__)
    app.register_blueprint(lstm_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(word2vec_bp)
    app.register_blueprint(chat_bp)
    return app