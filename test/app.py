import streamlit as st
from lastmodel import predict_unfairness, get_similar_clauses, generate_explanation

def main():
    st.set_page_config(
        page_title="약관 조항 불공정성 검사",
        page_icon="📃",
        layout="wide"
    )

    st.title("🔍 약관 조항 불공정성 검사기")
    st.markdown(
        """
        아래 텍스트 박스에 여러 약관 조항을 한 줄에 하나씩 입력하세요.  
        빈 줄을 입력하거나 '검사 시작' 버튼을 누르면, 각 조항이 불공정한지 여부를 판별하고  
        불공정 조항인 경우 유사 조항 예시와 함께 한국어 설명을 보여줍니다.
        """
    )

    # 사용자가 여러 줄로 입력할 수 있도록 text_area 사용
    input_text = st.text_area(
        label="📥 여러 약관 조항을 입력하세요 (한 줄에 하나의 조항)",
        placeholder="예시:\n제1조 이 계약은 …\n제2조 계약 해지 시 …\n…",
        height=200
    )

    # 입력된 텍스트를 줄 단위로 쪼개고, 빈 줄은 무시
    clauses = [line.strip() for line in input_text.split("\n") if line.strip() != ""]

    # 검사 시작 버튼
    if st.button("검사 시작"):
        if not clauses:
            st.warning("🔸 약관 조항을 한 줄씩 입력한 뒤, 다시 시도해 주세요.")
            return

        with st.spinner("✅ 약관 조항을 분석 중입니다... 잠시만 기다려 주세요."):
            # 1) 각 조항에 대해 불공정 여부를 예측
            labels = predict_unfairness(clauses)

            # 2) 결과를 페이지에 순서대로 출력
            for clause, label in zip(clauses, labels):
                st.markdown("---")
                st.write(f"**조항:** {clause}")

                if label == 1:
                    # 불공정 조항일 경우, 유사 조항 + 설명을 생성
                    similar = get_similar_clauses(clause, top_k=5)
                    explanation = generate_explanation(clause, similar)

                    st.error("⚠️ 불공정 조항으로 판단됨")
                    st.write("**유사 조항 예시:**")
                    for i, s in enumerate(similar, 1):
                        st.write(f"{i}. {s}")
                    st.write("**설명:**")
                    st.write(explanation)

                else:
                    # 공정 조항인 경우
                    st.success("✅ 공정(유리)한 조항으로 판단됨")

    # 사이드바에 간단한 사용법 추가
    with st.sidebar:
        st.header("ℹ️ 사용법")
        st.markdown(
            """
            1. ‘📥 여러 약관 조항을 입력하세요’ 칸에  
               한 줄에 하나씩 약관 조항을 입력합니다.  
            2. 입력을 마친 뒤, ‘검사 시작’ 버튼을 누르면 결과가 출력됩니다.  
            3. 불공정 조항으로 판단된 경우,  
               유사 조항 예시와 함께 설명을 확인할 수 있습니다.
            """
        )

if __name__ == "__main__":
    main()
