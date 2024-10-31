## 논문 요약, 질문&답변 챗봇
- **서비스:** 논문을 업로드하면 요약 및 논문의 중요 키워드를 보여주고 논문의 내용에 대한 질문을 하면 답을 해줍니다.
- **Streamlit application:** pdf 업로드 부분, 요약과 키워드를 보여주는 부분, 채팅 부분으로 구성되어 있습니다.
  논문 pdf를 읽어 올 때 문장들의 맥락에 대해 보존할 필요가 있을 것 같아 pymupdf4llm로 pdf를 읽고 langchain의 MarkdownTextSplitter로 텍스트를 쪼갰습니다.
- **LLM 선정:** 모델은 'gpt-4o-mini'을 사용했습니다.
