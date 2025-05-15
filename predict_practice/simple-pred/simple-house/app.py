import streamlit as st
import pickle
import numpy as np

# 모델 로드
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏠 집값 예측기")

# 사용자 입력 받기
lot_frontage = st.number_input("LotFrontage", value=60.0)
lot_area = st.number_input("LotArea", value=8000.0)
gr_liv_area = st.number_input("GrLivArea", value=1500.0)

# 예측 버튼
if st.button("예측하기"):
    input_data = np.array([[lot_frontage, lot_area, gr_liv_area]])
    prediction = model.predict(input_data)
    st.subheader(f"🏷️ 예측 집값: ${prediction[0]:,.2f}")
