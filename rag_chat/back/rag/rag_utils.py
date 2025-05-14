import os
import pickle
from langchain_community.vectorstores import FAISS # 검색용 벡터 저장소 (local 벡터 DB) 
from langchain.schema import Document # LangChain에서 문서를 담는 객체
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 문서를 여러 블록으로 자름
from langchain_huggingface import HuggingFaceEmbeddings # 문장을 벡터로 변환할 임베딩 모델
from dotenv import load_dotenv


# .env에서 환경변수 불러오기
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# 임베딩 모델 
embedding = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask"
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # back/
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_store.pkl")
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
# 벡터스토어를 저장할 파일명을 지정해놓음
#   DOC_PATH = os.path.join(os.path.dirname(__file__), "documents", "휴먼_2교육실_내부데이터.txt")


# 내부 문서 불러오기 
def load_documents():
    DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
    documents = []

    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))

    if not documents:
        raise FileNotFoundError(f"📂 documents 폴더에 .txt 파일이 없습니다: {DOCS_DIR}")
    
    return documents


def save_vectorstore(vectorstore, path=VECTORSTORE_PATH):
    with open(path, "wb") as f:
        pickle.dump(vectorstore, f)

# def load_vectorstore():
#     docs = load_documents()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30) # 문서 쪼개고 겹치도록 나눔 
#     splits = splitter.split_documents(docs)
#     return FAISS.from_documents(splits, embedding) 
#     # 각각의 split 문서를 embedding 모델로 최적화 , 벡터를 FAISS 에 저장 후 vectorstore 객체 반환 

def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH): # 이미 만들어둔 FAISS 벡터 DB 가 있으면 로드해서 재사용 
        with open(VECTORSTORE_PATH, "rb") as f:
            print("캐시된 vectorstore 로드됨")
            return pickle.load(f)
    else:
        print("vectorstore 새로 생성 중...")
        docs = load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        splits = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embedding)
        save_vectorstore(vectorstore)
        return vectorstore

def generate_rag_answer(model, question, vectorstore, k=4): 
    # 질문에 대해 벡터 유사도 기반으로 가장 비슷한 문서 k개 검색 
    docs = vectorstore.similarity_search(question, k=k) 
    context = "\n".join([f"문서{i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    prompt = f"""Context에 기반해서 짧게 대답해줘.
    Context:
    {context}
    질문: {question}
    대답:"""
    response = model.generate_content(prompt)
    answer = response.candidates[0].content.parts[0].text
    return answer, context
