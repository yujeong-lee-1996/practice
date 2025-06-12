
##  주요 프로젝트 설명

### 1. LSTM - 영화 리뷰 감성 예측
- **기술**: LSTM, 임베딩, 감정 분류
- **입력**: 텍스트 리뷰
- **출력**: 긍정 / 부정
- **특징**: 간단한 자연어처리 파이프라인부터 LSTM 기반 분류 실습

### 2. W2V - Word2Vec 벡터 연산
- **기술**: Word2Vec (Skip-gram / CBOW)
- **기능**: `왕 - 남자 + 여자 = 여왕`과 같은 유사 벡터 연산
- **활용**: 단어 간 의미적 유사성 시각화 및 검색

### 3. chatbot - Gemini 기반 챗봇
- **기술**: Google Gemini API, Flask, Prompt Engineering
- **모드**
  - 일반 모드: 일상 질의응답
  - 거짓말 모드: 의도적 비논리 응답
  - RAG 모드: 벡터DB 기반 근거 응답
- **특징**: LLM의 다양한 활용 방식을 비교 실험

### 4. my_web - 통합 웹 플랫폼
- **기술**: Flask, HTML/CSS, JS
- **기능**: LSTM, Word2Vec, 챗봇 기능 통합 UI 구현

#### 📂 my_web 디렉토리 구조

my_web/
├── data_files/ # 데이터 파일 (ex. 예시 입력, 임베딩 등)  
├── db/ # 데이터베이스 연동 모듈  
├── models/ # 학습된 모델 및 로딩 로직  
├── static/ # CSS, JS, 이미지 등 정적 리소스  
├── templates/ # HTML 템플릿 (Jinja2)  
├── train/ # 학습용 코드 (ex. LSTM 학습)  
├── utils/ # 유틸 함수 (전처리, 벡터 연산 등)  
├── views/ # Flask Blueprint 라우팅 분리  
├── .env # 환경 변수 (API 키 등)  
├── .gitignore # Git 무시 파일 설정  
├── init.py # Flask 앱 초기화  
└── run.py # Flask 앱 실행 진입점  

### 5. predict_practice - 경진대회 실습
- **내용**: Kaggle / Dacon 등 실전 예측 모델링
- **모델**: RandomForest, XGBoost, 딥러닝 등
- **목표**: 문제 정의부터 전처리, 모델 학습, 결과 분석까지 전 과정을 반복 실습


#### 가상환경 설치 및 실행
conda create -n python-env-311 python=3.11  
conda activate python-env-311

