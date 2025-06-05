#  ✅ views/board_view.py
import os
from flask import Blueprint, request, jsonify, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from db.board_util import insert_post,  get_posts_by_page, get_post_by_id, update_post, delete_post_by_id

board_bp = Blueprint('board', __name__, url_prefix='/project/board')

# # 라우트 정의
# @lstm_bp.route("/", methods=["GET", "POST"])
# def lstm_main():
#     prediction = None
#     if request.method == "POST":
#         input_text = request.form['text_input']
#         prediction = predict(input_text, model, device)

    # return render_template("lstm.html", prediction=prediction)

UPLOAD_FOLDER = 'static/uploads'


@board_bp.route('/')
def board_list():
    page = int(request.args.get('page', 1))  # 기본값 1
    per_page = 10

    posts, total = get_posts_by_page(page, per_page)
    total_pages = (total + per_page - 1) // per_page

    return render_template(
        'board/list.html',
        posts=posts,
        page=page,
        total_pages=total_pages
    )

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
        return render_template('board/write.html')

@board_bp.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    image_url = url_for('static', filename=f'uploads/{filename}')
    return jsonify({'url': image_url})

@board_bp.route('/post/<int:post_id>')
def post_detail(post_id):
    post = get_post_by_id(post_id)  # post 하나 조회 함수
    if post:
        return render_template('board/detail.html', post=post)
    else:
        return "해당 글이 존재하지 않습니다.", 404
    
@board_bp.route('/post/<int:post_id>/edit', methods=['GET'])
def edit_post(post_id):
    post = get_post_by_id(post_id)
    if not post:
        return "Post not found", 404
    return render_template('board/edit.html', post=post)


@board_bp.route('/post/<int:post_id>/edit', methods=['POST'])
def update_post_route(post_id):
    data = request.get_json()
    title = data.get('title')
    content = data.get('content')
    update_post(post_id, title, content)
    return jsonify({'message': '수정 완료!'})


@board_bp.route('/post/<int:post_id>/delete', methods=['POST'])
def delete_post(post_id):
    delete_post_by_id(post_id)
    return redirect(url_for('board.board_list'))