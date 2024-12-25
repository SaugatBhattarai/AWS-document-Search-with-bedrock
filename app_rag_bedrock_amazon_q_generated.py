#Document Search Using AWS BedRock, Langchain, streamlit and RAG
#Importing the required libraries
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms import Bedrock
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import boto3

#Initializing the Bedrock client
boto3.setup_default_session(region_name="us-west-2")
bedrock_client = boto3.client("bedrock-runtime")
bedrock_embedding = BedrockEmbeddings(client=bedrock_client,model_id="amazon.titan-embed-text-v1")

#Function to create the vector DB
def vector_embedding():
    loader = WebBaseLoader("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    docs = text_splitter.split_documents(document)
    db = FAISS.from_documents(docs,bedrock_embedding)
    return db

#Function to create the LLM
def bedrock_llm():
    bedrock_llm = Bedrock(client=bedrock_client,model_id="anthropic.claude-v2")
    return bedrock_llm

#Function to create the RAG chain
def rag_chain():
    db = vector_embedding()
    retriever = db.as_retriever(search_kwargs={"k":1})
    bedrock_llm = bedrock_llm()
    rag = RetrievalQA.from_chain_type(llm=bedrock_llm,chain_type="stuff",retriever=retriever)
    return rag

#Function to invoke the RAG chain
def rag_chain_invoke(question):
    rag = rag_chain()
    answer = rag.invoke(question)
    return answer['result']

#Streamlit code
st.title("Document Search Using AWS BedRock, Langchain, streamlit and RAG")
st.markdown("This application demonstrates the use of AWS Bedrock, Langchain, streamlit and RAG to search a document and answer questions about the document.")
question = st.text_input("Enter your question here")
if st.button("Search"):
    answer = rag_chain_invoke(question)
    st.write(answer)
    st.balloons()