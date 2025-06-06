### main.py를 웹에서 실행시키기 위해
### streamlit ui를 추가한 코드

import streamlit as st
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# 페이지 설정
st.set_page_config(
    page_title="계약서 조항 분석기",
    page_icon="📋",
    layout="wide"
)

st.title("📋 계약서 조항 유불리 분석기")
st.markdown("---")

@st.cache_resource
def load_models():
    """모델들을 로드하고 캐싱"""
    # 유불리 판단 모델
    classification_dir = "../model/classification"
    classifier_model = AutoModelForSequenceClassification.from_pretrained(classification_dir)
    classifier_tokenizer = AutoTokenizer.from_pretrained(classification_dir)
    
    # 의미 임베딩 모델
    semantic_dir = "../model/legal-kr-sbert-contrastive"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic_model = SentenceTransformer(semantic_dir).to(device)
    
    return classifier_model, classifier_tokenizer, semantic_model, device

@st.cache_data
def load_data():
    """데이터를 로드하고 캐싱"""
    dataset = torch.load(
        "embedding_dataset.pt",
        map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage
    )
    
    df = pd.read_csv("labeled.csv")
    
    return dataset, df

# 모델과 데이터 로드
with st.spinner("모델을 로드하는 중..."):
    classifier_model, classifier_tokenizer, semantic_model, device = load_models()
    dataset, df = load_data()
    
    texts = dataset["texts"]
    embeddings = dataset["embeddings"].to(device)

st.success(f"✅ 모델 로드 완료 (Device: {device})")

def predict_unfairness(clauses):
    """조항의 유불리를 예측"""
    inputs = classifier_tokenizer(clauses, padding=True, truncation=True, return_tensors="pt")
    outputs = classifier_model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1)
    return preds.tolist()  # 0: 유리, 1: 불리

def get_similar_clauses(query, top_k=5):
    """유사한 조항들을 검색"""
    query_emb = semantic_model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    for idx in top_results.indices:
        clause = texts[idx]
        # labeled.csv에서 basis 찾기
        matched = df[df["text"] == clause]
        basis = matched["basis"].values[0] if not matched.empty else ""
        similarity_score = cos_scores[idx].item()
        results.append({
            "clause": clause, 
            "basis": basis,
            "similarity": similarity_score
        })
    
    return results

@st.cache_resource
def setup_llm():
    """LLM 체인 설정"""
    llm = ChatOllama(model="anpigon/EEVE-Korean-10.8B:latest")
    
    prompt_template = PromptTemplate(
        input_variables=["clause", "similar"],
        template="""다음은 서비스 약관의 조항입니다:

조항:
{clause}

유사한 조항들:
{similar}

이 조항이 왜 불리한지 설명해 주세요.
"""
    )
    
    return LLMChain(llm=llm, prompt=prompt_template)

llm_chain = setup_llm()

def generate_explanation(clause, similar_clauses):
    """조항에 대한 설명 생성"""
    similar_text = "\n\n".join(
        [f"- 조항:\n{item['clause']}\n설명:\n{item.get('basis', '')}" for item in similar_clauses]
    )

    return llm_chain.run({
        "clause": clause,
        "similar": similar_text
    })

# UI 구성
st.markdown("## 📝 조항 입력")

# 입력 방식 선택
input_method = st.radio(
    "입력 방식을 선택하세요:",
    ["단일 조항", "여러 조항 (줄바꿈으로 구분)"]
)

if input_method == "단일 조항":
    clause_input = st.text_area(
        "분석할 조항을 입력하세요:",
        height=100,
        placeholder="예: 회사는 언제든지 사전 통지 없이 서비스를 중단할 수 있습니다."
    )
    input_clauses = [clause_input] if clause_input.strip() else []
else:
    multi_clause_input = st.text_area(
        "분석할 조항들을 입력하세요 (각 줄마다 하나씩):",
        height=200,
        placeholder="조항 1\n조항 2\n조항 3"
    )
    input_clauses = [line.strip() for line in multi_clause_input.split('\n') if line.strip()]

# 분석 설정
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("유사 조항 검색 개수", min_value=1, max_value=10, value=5)
with col2:
    min_similarity = st.slider("최소 유사도 임계값", min_value=0.0, max_value=1.0, value=0.5)

# 분석 실행
if st.button("🔍 조항 분석 시작", type="primary"):
    if not input_clauses:
        st.warning("분석할 조항을 입력해주세요.")
    else:
        with st.spinner("조항을 분석하는 중..."):
            labels = predict_unfairness(input_clauses)
            
            st.markdown("## 📊 분석 결과")
            
            for i, (clause, label) in enumerate(zip(input_clauses, labels), 1):
                with st.expander(f"조항 {i}: {'🔴 불리' if label == 1 else '🟢 유리'}", expanded=True):
                    st.markdown(f"**조항 내용:**")
                    st.info(clause)
                    
                    if label == 1:  # 불리한 조항
                        st.markdown("**🔍 유사 조항 분석:**")
                        
                        similar = get_similar_clauses(clause, top_k=top_k)
                        # 유사도 임계값 필터링
                        similar = [s for s in similar if s['similarity'] >= min_similarity]
                        
                        if similar:
                            # 유사 조항 표시
                            for j, sim_clause in enumerate(similar, 1):
                                with st.container():
                                    st.markdown(f"**유사 조항 {j} (유사도: {sim_clause['similarity']:.3f})**")
                                    st.text(sim_clause['clause'])
                                    if sim_clause['basis']:
                                        st.markdown("**근거:**")
                                        st.text(sim_clause['basis'])
                                    st.markdown("---")
                            
                            # AI 설명 생성
                            st.markdown("**🧠 AI 분석 설명:**")
                            with st.spinner("AI가 설명을 생성하는 중..."):
                                try:
                                    explanation = generate_explanation(clause, similar)
                                    st.markdown(explanation)
                                except Exception as e:
                                    st.error(f"설명 생성 중 오류가 발생했습니다: {str(e)}")
                        else:
                            st.warning(f"유사도 {min_similarity} 이상인 조항을 찾을 수 없습니다.")
                    
                    else:  # 유리한 조항
                        st.success("✅ 이 조항은 유리한 것으로 분석되었습니다.")

# 사이드바 정보
with st.sidebar:
    st.markdown("## ℹ️ 모델 정보")
    st.markdown(f"""
    - **분류 모델**: 조항 유불리 판단
    - **임베딩 모델**: 한국어 법률 SBERT
    - **생성 모델**: EEVE-Korean-10.8B
    - **처리 장치**: {device}
    - **데이터셋 크기**: {len(texts):,}개 조항
    """)
    
    st.markdown("## 🔧 사용 방법")
    st.markdown("""
    1. 분석할 계약서 조항을 입력
    2. 분석 설정 조정 (선택사항)
    3. '조항 분석 시작' 버튼 클릭
    4. 결과 확인 및 유사 조항 검토
    """)
    
    st.markdown("## ⚠️ 주의사항")
    st.markdown("""
    - AI 분석 결과는 참고용입니다
    - 법적 조언을 대체하지 않습니다
    - 전문가 검토를 권장합니다
    """)
