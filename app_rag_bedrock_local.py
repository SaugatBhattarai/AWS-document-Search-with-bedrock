""" 
    Document Search Using AWS BedRock, Langchain, streamlit and RAG 
"""
import json
import boto3
import os
import sys
import logging
import streamlit as st

# We will be using Titan Embeddings Model to generate embeddings for our documents and user queries
# from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Vector Embedding and Vector Store
from langchain_community.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# # Initialize logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(client=bedrock, model_id="amazon.titan-embed-text-v1")

#Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs


#Vector Embedding and Vector Store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings,
    )
    vectorstore_faiss.save_local("faiss_index")

#Create the Antropic Claude LLM
def get_claude_llm():
    try:
        claude_llm = ChatBedrock(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            client=bedrock,
            model_kwargs =  { 
                "max_tokens": 2048,
                "temperature": 0.9,
            }
        )
        return claude_llm
    except Exception as e:
        logger.error(f"Failed to initialize Llama2 LLM: {e}")
        raise

#Create the LLM Chain
prompt_template = """
Human: Use the following pieces of context to provide 
a concise answer the question at the end within 1500 words with details explanation. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Assistant:"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa.invoke({"query": query})  
    return answer['result']


def main():
    st.set_page_config("Document Search Using AWS Bedrock, Langchain, streamlit and RAG")
    st.header("Document Search Using AWS Bedrock")

    user_query = st.text_area("Ask a Question from the PDF Files.")
    
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Update Vector Store"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Search"):
        with st.spinner("Processing..."):
            faiss_index=FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            claude_llm = get_claude_llm()
            st.write(get_response_llm(claude_llm, faiss_index, user_query))
            st.success("Done")


if __name__ == "__main__":
    main()