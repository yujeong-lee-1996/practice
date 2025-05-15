import os
import pickle
from langchain_community.vectorstores import FAISS # 검색용 벡터 저장소 (local 벡터 DB) 
from langchain.schema import Document # LangChain에서 문서를 담는 객체
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 문서를 여러 블록으로 자름
from langchain_huggingface import HuggingFaceEmbeddings # 문장을 벡터로 변환할 임베딩 모델
from dotenv import load_dotenv
from keybert import KeyBERT


# 환경 변수 로딩 및 HuggingFace API 설정
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

kw_model = KeyBERT(model="paraphrase-MiniLM-L6-v2")

# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# 임베딩 모델 설정 
embedding = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask"
)

# 경로 설정 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # back/
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_store.pkl") # 벡터DB 캐시 저장 위치 
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents") # documents/*.txt: 텍스트 문서를 읽어올 디렉토리



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
                # 파일 을 읽어 LangChain 용 Document 객체 리스트로 반환 

    if not documents:
        raise FileNotFoundError(f"📂 documents 폴더에 .txt 파일이 없습니다: {DOCS_DIR}")
    
    return documents

# FAISS 로 만든 벡터 저장소르르 .pkl 파일로 로컬 저장. 
def save_vectorstore(vectorstore, path=VECTORSTORE_PATH):
    with open(path, "wb") as f:
        pickle.dump(vectorstore, f)

# def load_vectorstore():
#     docs = load_documents()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30) # 문서 쪼개고 겹치도록 나눔 
#     splits = splitter.split_documents(docs)
#     return FAISS.from_documents(splits, embedding) 
#     # 각각의 split 문서를 embedding 모델로 최적화 , 벡터를 FAISS 에 저장 후 vectorstore 객체 반환 

# 핵심 키워드 추출 함수
def extract_keywords_by_keybert(text, top_n=3):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=top_n)
    return [kw for kw, _ in keywords]

def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH): # 이미 만들어둔 FAISS 벡터 DB 가 있으면 로드해서 재사용 
        with open(VECTORSTORE_PATH, "rb") as f:
            print("캐시된 vectorstore 로드됨")
            return pickle.load(f)
    else:
        print("vectorstore 새로 생성 중...")
        docs = load_documents() # 문서로딩 
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30) # 텍스트 분할 
        splits = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embedding) # 벡터화 
        save_vectorstore(vectorstore) # 저장 
        return vectorstore

# 답변 생성 함수 
def generate_rag_answer(model, question, vectorstore, k=4): 

    # 키워드 추룰 
    keywords = extract_keywords_by_keybert(question)
    # 키워드 기반 검색
    query = " ".join(keywords)

    # 질문에 대해 벡터 유사도 기반으로 가장 비슷한 문서 k개 검색 
    docs = vectorstore.similarity_search(query, k=k) 

    context = "\n".join([f"문서{i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    prompt = f"""Context에 기반해서 짧게 대답해줘.
    Context:
    {context}
    질문: {question}
    대답:"""
    response = model.generate_content(prompt)
    answer = response.candidates[0].content.parts[0].text
    return answer, context
