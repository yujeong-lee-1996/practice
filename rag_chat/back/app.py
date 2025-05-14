from flask import Flask, send_from_directory
from flask_cors import CORS
from routes.chat import chat_bp

app = Flask(__name__, static_folder="../front", static_url_path="")
CORS(app)

app.register_blueprint(chat_bp)

@app.route("/")
def serve_index():
    return send_from_directory("../front", "index.html")

if __name__ == "__main__":
    app.run(debug=True)