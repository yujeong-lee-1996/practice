import os
from flask import Blueprint, request, jsonify
import google.generativeai as genai
from rag.rag_utils import load_vectorstore, generate_rag_answer
from dotenv import load_dotenv

chat_bp = Blueprint("chat", __name__, url_prefix="/api")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

chat_history = {
    "chat": [],
    "lie": [],
    "rag": []
}

def update_history(mode, role, message):
    chat_history[mode].append({"role": role, "message": message})

@chat_bp.route("/chat", methods=["POST"])
def chat():
    question = request.json.get("message", "")
    update_history("chat", "user", question)

    response = model.generate_content(question)
    answer = response.text
    update_history("chat", "bot", answer)

    return jsonify({"response": answer, "history": chat_history["chat"]})

@chat_bp.route("/lie", methods=["POST"])
def lie():
    question = request.json.get("message", "")
    update_history("lie", "user", question)
 
    # prompt -> Gemini 같은 생성형 AI에게 "어떻게 대답할지"를 지시하는 입력 문자열 
    prompt = f"""너는 거짓말쟁이 챗봇이야. 항상 거짓말을해.
    모든 질문에 대해 반드시 사실이 아닌 거짓된 정보를 창의적으로 대답해야 해!
    절대 실제 정보나 진실을 말하면 안 돼. 반드시 틀린 정보를 제공해야 해.
    질문: {question}
    대답:"""
    response = model.generate_content(prompt)
    answer = response.text
    update_history("lie", "bot", answer)

    return jsonify({"response": answer, "history": chat_history["lie"]})

@chat_bp.route("/rag", methods=["POST"])
def rag():
    question = request.json.get("message", "")
    update_history("rag", "user", question)

    vectorstore = load_vectorstore()
    answer, context = generate_rag_answer(model, question, vectorstore)
    update_history("rag", "bot", answer)

    return jsonify({"response": answer, "context": context, "history": chat_history["rag"]})
