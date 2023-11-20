import pyodbc
import streamlit as st
import openai
import pandas as pd
import os

with st.sidebar:
    st.title('🗨️ Natural Language -- SQL Query Chatbot')
    st.markdown('''
    ## About App:

    To create the app, The primary resources utilised are:

    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)
    - [Adventure Works Database](https://learn.microsoft.com/en-us/sql/samples/adventureworks-install-configure?view=sql-server-ver16&tabs=ssms)

    ## About me:
    - [Sanjeev Malik Github](https://github.com/Malikk1997) All code here
    - [Linkedin](https://www.linkedin.com/in/sanjeev-malik-41a545170/)
    
    ''')
    st.write("Made by Sanjeev Malik")

def main():
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.header("SQL Database querying in Natural Language")
        # Connect SQL Server with python with paramters - Driver, Server Name, Database Name & Trusted Connection.
        
        conn_str = "DRIVER={ODBC Driver 17 for SQL Server};Server=DESKTOP-S2KR9DN;Database=AdventureWorks2019;trusted_connection=yes;"
        conn = pyodbc.connect(conn_str, timeout = 0)
        cursor = conn.cursor()
        
        # # Test the connection using a random query
        # cursor.execute('SELECT * FROM [Person].[Person]')
        # rows = cursor.fetchall()
        # for row in rows:
        #     print (row)

        # Execute a SQL Query to extract all the Table Names
        cursor.execute("SELECT concat(Table_schema,'.',Table_name) FROM information_schema.tables")
        tables_list = cursor.fetchall()
        tables_list = [str(i).strip('(').strip(')').strip(",") for i in tables_list]

        # Create a SelectBox with options of extracted Table Names
        option=st.selectbox("Select Your Table: ",options=tables_list)
        option=str(option).strip('(').strip(')').strip(',').strip("'")

        # Display the Selected Table Name
        st.markdown(f"#### Table Chosen: {option}")
        table_name=str(option).split(".")[1].strip("'")

        # Execute a Query to extract all the columns from the selected Table
        cursor.execute(f"SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'")
        columns = cursor.fetchall()
        columns = [str(i).strip('(').strip(')').strip(",")[1:-1] for i in columns]

        # Diplay all the columns from the Table
        st.write(f" #### Columns in the Table:")
        for column in columns:
            st.write(f"- {column}")
        
        # Read the user query 
        query=st.text_input("Write your query") 

        if query:
            # With the help of prompt template Generate an SQL Query with custom inputs using Natural Language.
            prompt=f"Generate a SQL server query to retrive {query} from the database with table {option} "
            response=openai.Completion.create(api_key=os.environ['OPENAI_API_KEY'], engine='text-davinci-003',prompt=prompt,max_tokens=100) 
            generated_query=response.choices[0].text.strip()

            # Display & Execute the Generated SQL Query 
            st.write(f"Generated SQL Query: {generated_query}")
            result=cursor.execute(generated_query)
            res_data = result.fetchall()

            # Display the data in Tabular Format using DataFrame
            data=pd.DataFrame(res_data)          
            data[0]=data[0].astype(str)
            
            # Split the data into its respective columns
            columns = data[0].str.split(',', expand=True)
            data = pd.DataFrame(columns)
            st.write(data)


if __name__ == '__main__':
    main()