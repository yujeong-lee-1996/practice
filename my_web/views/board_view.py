#  ✅ views/board_view.py
import os
from flask import Blueprint, request, jsonify, render_template, current_app
from werkzeug.utils import secure_filename
from db.board_util import insert_post

board_bp = Blueprint('board', __name__, url_prefix='/project/board')

# # 라우트 정의
# @lstm_bp.route("/", methods=["GET", "POST"])
# def lstm_main():
#     prediction = None
#     if request.method == "POST":
#         input_text = request.form['text_input']
#         prediction = predict(input_text, model, device)

    # return render_template("lstm.html", prediction=prediction)


@board_bp.route('/write', methods=['GET', 'POST'])
def write_post():
    if request.method == 'POST':
        data = request.get_json()
        print('data',data)
        title = data.get('title')
        content = data.get('content')
        insert_post(title, content)
        return jsonify({'message': '작성 완료!'}), 200
    else:
        return render_template('board.html')

@board_bp.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        url = f'/static/uploads/{filename}'
        return jsonify({"url": url})
    return jsonify({'error': 'No file'}), 400
