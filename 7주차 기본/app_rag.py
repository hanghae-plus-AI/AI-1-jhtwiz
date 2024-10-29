from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

load_dotenv()

url = 'https://spartacodingclub.kr/blog/all-in-challenge_winner'

loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("editedContent")
        )
    ),
)
docs = loader.load()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"
    st.session_state["max_tokens"] = 512  # 최대 출력 token 수
    st.session_state["temperature"] = 0.1  # sampling할 때 사용하는 temperature
    st.session_state["frequency_penalty"] = 0.0  # 반복해서 나오는 token들을 조절하는 인자

st.title("GPT Bot")

llm = ChatOpenAI(model=st.session_state["openai_model"],
                 max_tokens=st.session_state["max_tokens"],
                temperature=st.session_state["temperature"],
                frequency_penalty=st.session_state["frequency_penalty"])

if user_text := st.chat_input("질문을 입력하세요"):
    with st.chat_message("user"):
        st.markdown(user_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings(),
        )
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(user_text)
        
        prompt = hub.pull("rlm/rag-prompt")
        user_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": user_text})

    with st.chat_message("assistant"):
        response_stream = llm.stream(user_prompt)
        response = st.write_stream(response_stream)