import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# 유불리 판단 모델 로드
classifier_model = AutoModelForSequenceClassification.from_pretrained("./test/classification_model")
classifier_tokenizer = AutoTokenizer.from_pretrained("./test/classification_model")

# 의미 임베딩 모델 로드
semantic_model = SentenceTransformer("./test/legal-kr-sbert-contrastive")

# 임베딩 데이터셋 로드 (CPU 매핑)
dataset = torch.load("./test/embedding_dataset.pt", map_location=torch.device('cpu'))
texts = dataset["texts"]
embeddings = dataset["embeddings"]

# labeled.csv 읽기
df = pd.read_csv("./test/labeled.csv")

def predict_unfairness(clauses):
    inputs = classifier_tokenizer(clauses, padding=True, truncation=True, return_tensors="pt")
    outputs = classifier_model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1)
    return preds.tolist()  # 0: 유리, 1: 불리

def get_similar_clauses(query, top_k=5):
    query_emb = semantic_model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    for idx in top_results.indices:
        clause = texts[idx]
        matched = df[df["text"] == clause]
        basis = matched["basis"].values[0] if not matched.empty else ""
        results.append({"clause": clause, "basis": basis})
    
    return results

# Ollama LLM 설정
llm = ChatOllama(model="anpigon/EEVE-Korean-10.8B:latest")

# 프롬프트 템플릿
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

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def generate_explanation(clause, similar_clauses):
    similar_text = "\n\n".join(
        [f"- 조항:\n{item['clause']}\n설명:\n{item.get('basis', '')}" for item in similar_clauses]
    )
    return llm_chain.run({
        "clause": clause,
        "similar": similar_text
    })
