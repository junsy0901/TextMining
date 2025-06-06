# 계약서 조항 유불리 분석기
한국어 계약서 조항을 입력하면, 해당 조항이 유리한지 불리한지를 판단하고, 불리한 조항일 경우 유사한 기존 조항들을 검색하여 그 근거를 바탕으로 왜 불리한지 설명해주는 AI 기반 분석 도구입니다.

## 🚀 주요 기능
조항 유불리 판단: 딥러닝 분류 모델을 통한 자동 조항 분류

유사 조항 검색: 한국어 법률 텍스트 임베딩을 활용한 의미 기반 검색

AI 설명 생성: 유사 조항의 근거를 바탕으로 한 상세 설명 제공

직관적 UI: Streamlit 기반 사용자 친화적 인터페이스

## 📦 설치 및 실행
1. 환경 설정
```bash
# Conda 가상환경 생성 및 활성화
conda create -n textmining python=3.8 -y
conda activate textmining
```
2. 의존성 설치
```bash
pip install -r requirements.txt
```
3. 필요 파일 준비
프로젝트 루트 디렉토리에 다음 파일들이 필요합니다:

embedding_dataset.pt: 임베딩된 조항 데이터셋

labeled.csv: 라벨링된 조항과 근거 데이터

../model/classification/: 분류 모델 디렉토리

../model/legal-kr-sbert-contrastive/: 임베딩 모델 디렉토리

4. 애플리케이션 실행
```bash
streamlit run app.py
```
📋 requirements.txt
```text
streamlit
torch
sentence-transformers
transformers
langchain
langchain-community
pandas
```
