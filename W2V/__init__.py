from flask import Flask
from flask import render_template

from .views import main_view, word2vec_view 



def create_app(): # Flask가 웹 애플리케이션을 시작할 때 자동으로 호출하는 함수

    app = Flask(__name__) # web application 만들기

    app.config['SECRET_KEY'] = 'humanda5-secret-key' # 세션(session) 등을 사용하기 필요한 설정

    # @app.route("/")
    # def index():
    #     return render_template('index.html')

    app.register_blueprint(main_view.main_bp)
    app.register_blueprint(word2vec_view.word2vec_bp)


    return app