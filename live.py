import streamlit as st
import nest_asyncio
import logging
import os


from typing import List
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.llm.openai import OpenAIChat
from phi.knowledge import AssistantKnowledge
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
from dotenv import load_dotenv
from Database import db_url



load_dotenv()
os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)







def setup_assistant(llm: str) -> Assistant:
    return Assistant(
        name="auto_rag_assistant",
        llm=llm,
        storage=PgAssistantStorage(table_name="auto_rag_assistant_openai", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="auto_rag_documents_openai",
                embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
            ),
            num_documents=3,
        ),
        description="You are a helpful Assistant called 'AutoRAG' and your goal is to assist the user in the best way possible.",
        instructions=[
            "Given a user query, first ALWAYS search your knowledge base using the `search_knowledge_base` tool to see if you have relevant information.",
            "If you don't find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If you need to reference the chat history, use the `get_chat_history` tool.",
            "If the user's question is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        tools=[DuckDuckGo()],
        markdown=True,
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=True,
    )


def add_document_to_kb(assistant:Assistant,file_path:str,file_type:str = "pdf"):
    if file_type == "pdf":
        reader = PDFReader()
    elif file_type == "url":
        reader = WebsiteReader()
    else:
        raise ValueError("Unsupported File Error")
    
    documents: List[Document] = reader.read(file_path)

    if documents:
        assistant.knowledge_base.load_documents(documents,upsert=True)
        logger.info(f"Document '{file_path}' added to the knowledge base.")
    else:
        logger.error("Could not read Document")





def query_assistant(assistant:Assistant,question:str):
    response = ""
    
    for delta in assistant.run(question):
        response +=  delta
    return response




def search_knowledge_base(assistant:Assistant,search_query:str):
    search_results = assistant.knowledge_base.search(search_query)
    return search_results



def search_internet(assistant:Assistant,search_query:str):
    search_query = f"duckduckgo_search:{search_query}"
    response = ""
    for delta in assistant.run(search_query):
        response += delta
    return response


def main():
    nest_asyncio.apply()
    st.title("AutoRAG Assistant")

    # Model selection
    model_option = st.selectbox(
        "Select the model",
        ("gpt-3.5-turbo", "gpt-4", "gpt-4o")
    )
    
    llm_model = os.getenv("OPENAI_MODEL_NAME", model_option)
    llm = OpenAIChat(model=llm_model)
    assistant = setup_assistant(llm)

    st.header("Add Document to Knowledge Base")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        add_document_to_kb(assistant, "temp.pdf", file_type="pdf")
        st.success("PDF added to the knowledge base.")

    
    url = st.text_input("Enter a URL to add to the knowledge base")
    if st.button("Add URL"):
        if url:
            add_document_to_kb(assistant, url, file_type="url")
            st.success("URL content added to the knowledge base.")
        else:
            st.error("Please enter a valid URL.")

    st.header("Ask a Question")
    query = st.text_input("Enter your question:")
    if st.button("Ask"):
        response = query_assistant(assistant, query)
        st.write("Query:", query)
        st.write("Response:", response)

    st.header("Search")
    search_query = st.text_input("Enter search query:")
    search_option = st.radio("Select search option", ("Knowledge Base", "Internet"))

    if st.button("Search"):
        if search_option == "Knowledge Base":
            search_results = search_knowledge_base(assistant, search_query)
            if search_results:
                st.write("Knowledge Base Search Results:")
                for result in search_results:
                    st.write(result.content)
            else:
                st.write("No results found in the knowledge base.")
        elif search_option == "Internet":
            search_results = search_internet(assistant, search_query)
            if search_results:
                st.write("Internet Search Results (DuckDuckGo):")
                st.write(search_results)
            else:
                st.write("No results found on the internet.")

if __name__ == "__main__":
    main()








