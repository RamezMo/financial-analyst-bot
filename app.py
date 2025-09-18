import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- App Configuration ---
st.set_page_config(page_title="Financial Analyst Bot", page_icon="🤖")
st.title("🤖 Conversational Financial Analyst")
st.info(
    """
    **Welcome! I'm your AI Financial Analyst.**

    * **Ask me anything** about the pre-loaded **Microsoft 2024 Annual Report** right now.
    * Or, **upload your own PDF** using the sidebar to start a new analysis.
    """
)
# --- Constants ---
DEFAULT_INDEX_PATH = "microsoft_faiss_index"

# --- Core RAG & Caching Functions ---

def process_uploaded_pdf(uploaded_file):
    """Loads and splits the uploaded PDF document into chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    os.remove(tmp_file_path) # Clean up the temporary file
    return chunks

def create_vector_store_from_chunks(chunks):
    """Creates a FAISS vector store in memory using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

@st.cache_resource
def get_retriever(uploaded_file=None):
    if uploaded_file:
        with st.status(f"Processing '{uploaded_file.name}'...", expanded=True) as status:
            status.update(label="Loading and splitting document...")
            chunks = process_uploaded_pdf(uploaded_file)
            
            status.update(label="Creating vector store with OpenAI embeddings...")
            vector_store = create_vector_store_from_chunks(chunks)
            
            status.update(label="Processing Complete!", state="complete", expanded=False)
        return vector_store.as_retriever()
    
    elif os.path.exists(DEFAULT_INDEX_PATH):
        st.info(f"Loading knowledge base from '{DEFAULT_INDEX_PATH}'...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local(DEFAULT_INDEX_PATH, embeddings, allow_dangerous_deserialization=True).as_retriever()
    
    return None

def get_conversational_rag_chain(retriever):
    """Creates the main conversational RAG chain."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # 1. Contextualize Question Chain: This part rephrases the user's question.
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # 2. Answering Chain: This part answers the (now complete) question.
    qa_system_prompt = """    You are a professional financial analyst AI. Use the following pieces of retrieved context to answer the user's question. \
      If the answer is not in the context but the question is similar to another question from the doc then show the correct question and ask if he want the answer. \
      If the user greeting you answer him in an intelligent way, otherwise state that the information is not available in the document. \
      If you don't know the answer and no similar question, just say that you don't know. Use three sentences maximum and keep the answer concise.\
      do not view the source if you do not answer the question the user sent from the doc\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Combine them into the final chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# --- Sidebar ---
with st.sidebar:
    st.header("Your Document")
    uploaded_file = st.file_uploader("Upload a PDF to begin analysis", type=['pdf'])

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

retriever = get_retriever(uploaded_file)
if retriever:
    # Reset chat history if it's a new file upload or the first run with the default index
    is_new_upload = uploaded_file and st.session_state.uploaded_file_name != uploaded_file.name
    is_first_load = st.session_state.rag_chain is None

    if is_new_upload:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.messages = [AIMessage(content=f"Ready to chat about {uploaded_file.name}.")]
        st.session_state.rag_chain = get_conversational_rag_chain(retriever)
    elif is_first_load:
         st.session_state.messages = [AIMessage(content="Ready to answer questions about the pre-loaded Microsoft report.")]
         st.session_state.rag_chain = get_conversational_rag_chain(retriever)

elif not retriever and not uploaded_file and not os.path.exists(DEFAULT_INDEX_PATH):
    st.warning("Default knowledge base not found. Please upload a document to begin.")


# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("Ask a question...", disabled=(st.session_state.rag_chain is None)):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # We pass the entire chat history (except the last question) to the chain
            chat_history = st.session_state.messages[:-1]
            response = st.session_state.rag_chain.invoke({"input": prompt, "chat_history": chat_history})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append(AIMessage(content=answer))
    st.rerun()