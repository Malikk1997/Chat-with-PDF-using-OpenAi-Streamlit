import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title(' ♠️ LLM Chat App ♠️ ')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [Sanjeev Malik Github](https://github.com/Malikk1997) All code here
    ''')
    add_vertical_space(5)
    st.write('Made by [Sanjeev Malik ♠️♠️]')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # Initialize session_state if not exists
    if "session_state" not in st.session_state:
        st.session_state.session_state = {}

    # Get or create conversation in session_state
    conversation = st.session_state.session_state.get("conversation", [])
    st.session_state.session_state["conversation"] = conversation

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        # Load or create VectorStore
        vector_store_path = f"{store_name}.pkl"
        if os.path.exists(vector_store_path):
            with open(vector_store_path, "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            with open(vector_store_path, "wb") as f:
                pickle.dump(VectorStore, f)

        openai_api_key = os.environ["OPENAI_API_KEY"]
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            
            st.write(response)

            # Update and display conversation
            conversation.append({"role": "user", "content": query})
            conversation.append({"role": "assistant", "content": response})
            st.session_state.session_state["conversation"] = conversation

            st.subheader("Current Conversation:")
            for message in conversation:
                st.write(f"{message['role']}: {message['content']}")

if __name__ == '__main__':
    main()
