import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain



from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import requests

from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)



openai_api_key = os.environ["OPENAI_API_KEY"]





def get_chat_prompt():
    template = """
  Act like an interviewer.

Ask the user for their name.
Once they enter their name, greet them with "Nice to meet you! Let us begin the interview." 
Also, ask them to please introduce yourself.
Once the user gives the introduction, ask them what is your domain.
Ask one question at a time related to the domain they mentioned after user answering, go for the  next question.
If the user doesn't know the answer, proceed to the next question.
After asking 3 domain-related questions, say "Thank you for your time. You have done well! All the best."
ask only one question at a time

{chat_history}
    """

    system_message = SystemMessagePromptTemplate.from_template(template)
    human_message = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)
    place_holder = MessagesPlaceholder(variable_name="chat_history")
    return ChatPromptTemplate.from_messages([system_message, place_holder, human_message_prompt])


def main():
    st.header("Interviewing chat bot")

    message_placeholder = st.container()
    message_placeholder.markdown('Robot:   ' + "Hi I am here to take your interview.please provide your name")


    # Initialize Streamlit chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def user(query):
        message_placeholder.markdown('User:   ' + query)

    def bot(response):
        message_placeholder.markdown('Robot:   ' + response)

    with st.form('chat_input_form'):

        query = st.text_input("   ", placeholder="   ")

        submit = st.form_submit_button("Submit")

    if st.button("New Chat"):
        st.session_state.clear()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "User":
            message_placeholder.markdown('User:   ' + message["content"])
        else:
            message_placeholder.markdown('Robot:   ' + message["content"])

    if submit:

        st.session_state.messages.append({"role": "User", "content": query})
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chat_prompt = get_chat_prompt()
        chain = ConversationChain(llm=llm, prompt=chat_prompt, memory=memory)

        for chat in st.session_state.chat_history:
            memory.save_context({"input": chat["input"]}, {"output": chat["output"]})

        user(query)

        with get_openai_callback() as cb:

            response = chain.run(query)
            print(cb)

        bot(response)

        st.session_state.messages.append({"role": "ai", "content": response})
        st.session_state.chat_history.append({"input": query, "output": response})



if __name__ == '__main__':
    main()