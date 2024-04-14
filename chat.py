from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PIL import Image
import base64

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
img = Image.open(r"C:\Users\ADMIN\Desktop\PYTHON CODES\Streamlit Projects\chatpdf\chatpng.png")
st.set_page_config(page_title="Mugare Chat: Document Generation AI", page_icon=img)

# Function to convert image to base64
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Define sidebar content
with st.sidebar:
    st.title("LLM Chatapp using LangChain ")
    st.markdown('''
        This app is an LLM powered Chatbot that answers questions based on Uploaded PDF.
        Here are some questions that you can ask: What is AI?
    ''')

# Custom CSS for glowing effect
st.markdown(
    """
    <style>
    .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #0066ff,
            0 0 10px #0066ff,
            0 0 15px #0066ff,
            0 0 20px #0066ff,
            0 0 25px #0066ff,
            0 0 30px #0066ff,
            0 0 35px #0066ff;
        position: relative;
        z-index: -1;
        border-radius: 30px;  /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to convert GIF image to base64
def gif_to_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

# Load and display sidebar image with glowing effect
def load_gif_to_sidebar(gif_path):
    gif_base64 = gif_to_base64(gif_path)
    st.sidebar.markdown(
        f'<img src="data:image/gif;base64,{gif_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

# Load GIF to sidebar
gif_path = "imgs/chatbot2.gif"
load_gif_to_sidebar(gif_path)

# Main content
st.header("Ask Your PDF📄")
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )  

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    query = st.text_input("Ask your Question about your PDF")
    if query:
        docs = knowledge_base.similarity_search(query)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
           
        st.success(response)
