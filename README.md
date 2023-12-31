# Chat-with-PDF-using-OpenAi-Streamlit
Here, I am working on this project and trying to upgrade its functionality day by day. So, I am going to split my work so as to make you understand it better where I added new functionality like Version 1, 2 and so on. And how it increases the efficiency. Also, it will help to understand smaller chunks bit by bit. 

This chat_pdf_code.py file is the First Task.

Task 1 & 2- Development of a Multi-Turn Streamlit Chatbot app for PDF-Based Q&A with Context Preservation.
Problem Statement:
Develop a Streamlit-based chatbot application that empowers users to interact with textual content from uploaded PDF documents using the OpenAI API. The primary goal of this application is to facilitate seamless and interactive information retrieval, enabling users to engage in multi-turn conversations with the chatbot while preserving context and memory of previous interactions.
 
Functional Requirements:
 
1.PDF Upload: The application should allow users to upload PDF documents containing textual content.
 
2.OpenAI Integration: Utilize the OpenAI API to process the uploaded PDF documents and extract key information from them.
 
3.Question-Answer Interaction: Users should be able to ask questions related to the content of the PDF document, and the chatbot must respond with relevant answers. The chatbot should understand and process natural language queries.
 
4.Multi-turn Conversation: The chatbot should maintain the context of the conversation, allowing users to ask follow-up questions or refer back to previous responses. This ensures a coherent and natural interaction.
 
5.Context Preservation: The chatbot should have memory of past questions and answers, maintaining context and facilitating meaningful dialogues.
 
Optional Task: PDF Document Cache
 
Consider implementing a caching mechanism to remember previously uploaded PDF documents for each user. This cache would prevent the need to re-upload the same PDFs unnecessarily and enhance the user experience.

OUTPUT:
I have built web application using streamlit and one can drag and drop the pdf and question on it like we did in chatgpt. Also, it will preserve the previous chats and cache.


Task 6-
Problem Statement: Building a Streamlit Chatbot App for Conducting User Interviews.
 
Description: The goal of this project is to develop a Streamlit chatbot application that conducts user interviews. The chatbot will prompt users to introduce themselves and specify the job role they are seeking an interview for (e.g., HR manager, Developer, Project Manager, etc.). Based on the specified job role, the chatbot will ask 3-4 role-specific questions. The follow-up questions will be tailored based on the user's responses. After the interview questions, the chatbot will conclude the interview by providing a brief feedback on the user's performance. Additionally, the application will include an "exit chat" button to clear the conversation, allowing a new user to start a new interview.
  
Techniques and Libraries: Utilize Langchain and OpenAi api to generate natural language responses and feedback based on user interactions. Implement prompt templates to structure the interview process and guide the conversation flow. Leverage Streamlit for building the user interface and handling the interview process.
