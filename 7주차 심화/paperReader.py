from dotenv import load_dotenv
import streamlit as st
import pymupdf4llm
import tempfile
import chromadb
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

load_dotenv()

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def render_chat(docs):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
                
    if user_text := st.chat_input("질문을 입력하세요"):
        with st.chat_message("user"):
            st.markdown(user_text)
            vectordb = Chroma.from_documents(
                documents=docs,
                embedding=OpenAIEmbeddings(),
            )
            retriever = vectordb.as_retriever()
            retrieved_docs = retriever.invoke(user_text)            
            prompt = hub.pull("rlm/rag-prompt")
            user_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": f"{user_text}. 주의: 답변은 반드시 한국어로 작성해야 합니다.**중요 지침:** 주어진 내용과 연관되지 않은 질문에 대해서는 절대 답변하지 말고, 질문이 제공된 내용과 관련된 경우에만 답변을 작성해 주세요."})
        
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=512, temperature=0.1)
            response_stream = llm.stream(user_prompt)
            response = st.write_stream(response_stream)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response
        })
        
        
def get_docs_with_summarize(pdf):
    docs = None
    
    if st.session_state.needUpdate:
        with st.status("AI가 논문을 읽고 있습니다...", expanded=True) as status:
            md_text = pymupdf4llm.to_markdown(pdf)
            splitter = MarkdownTextSplitter(chunk_size=4000, chunk_overlap=200)
            docs = splitter.create_documents([md_text])
            st.session_state.docs = docs
            llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024, temperature=0.4)
            template = '''다음의 내용을 한글로 요약해줘. 또한 가장 중요한 키워드를 5개 이내의 리스트로 작성해줘. *주의: 학술 용어는 번역하지 않아도 좋습니다.
            {text}
            '''
            st.write('AI가 논문을 요약하고 있습니다. 잠시만 기다려주세요!')
            prompt = PromptTemplate(template=template, input_variables=['text'])
            summary_chain = load_summarize_chain(llm, chain_type="map_reduce", combine_prompt=prompt)
            summary = summary_chain.run(input_documents=docs)
            status.update(label="요약이 완료되었습니다. 논문에 대한 질문은 하단 채팅 영역에서 진행하세요", state="complete", expanded=False)
            
        st.session_state.summary = summary
        st.write(summary)
    elif 'summary' in st.session_state:
        docs = st.session_state.docs
        summary = st.session_state.summary
        st.write(summary)
    
    return docs
    
    
def main():
    st.title("논문 요약 + 질문&답변 봇") 
    pdf = st.file_uploader("논문 PDF를 업로드해주세요", type="pdf") 
    temp_pdf_path = None
    if pdf is not None and pdf != st.session_state.pdf:
        st.session_state.pdf = pdf
        st.session_state.needUpdate = True
        st.session_state.messages = []
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(pdf.read())
            temp_pdf_path = temp_pdf.name
               
    docs = get_docs_with_summarize(temp_pdf_path)
    
    if docs: render_chat(docs)


if __name__ == '__main__':
    st.session_state.needUpdate = False
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    if 'pdf' not in st.session_state:
        st.session_state.pdf = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    main()