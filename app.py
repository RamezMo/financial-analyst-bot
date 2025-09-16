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
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- App Configuration ---
st.set_page_config(page_title="Financial Analyst Bot", page_icon="🤖")
st.title("🤖 Financial Analyst Q&A Bot")
st.info("This app can answer questions about the pre-loaded Microsoft 2024 Annual Report, or you can upload your own document in the sidebar.")

# --- Constants ---
DEFAULT_INDEX_PATH = "microsoft_faiss_index"

# --- Core RAG Functions ---
@st.cache_resource
def get_retriever(uploaded_file=None):
    """Creates a retriever from a PDF file or a pre-existing index."""
    if uploaded_file:
        with st.status(f"Processing '{uploaded_file.name}'...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            status.update(label="Chunking document...")
            loader = PyPDFLoader(tmp_file_path)
            chunks = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
            
            status.update(label="Creating embeddings with OpenAI...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            os.remove(tmp_file_path)
            status.update(label="Processing Complete!", state="complete", expanded=False)
        return vector_store.as_retriever()
    
    elif os.path.exists(DEFAULT_INDEX_PATH):
        st.info(f"Loading knowledge base from '{DEFAULT_INDEX_PATH}'...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.load_local(DEFAULT_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store.as_retriever()
    
    return None

def get_rag_chain(retriever):
    """Creates the RAG chain that returns sources for document-specific questions."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = PromptTemplate.from_template("You are a professional financial analyst AI. Answer the user's question based ONLY on the following context. If the answer is not in the context, state that the information is not available in the document.\n\nContext:\n{context}\n\nQuestion: {question}")
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=(RunnablePassthrough.assign(context=(lambda x: x["context"]))| prompt | llm | StrOutputParser()))
    )
    return chain

def get_conversational_chain():
    """Creates a simple conversational chain for greetings and chit-chat."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = PromptTemplate.from_template("You are a friendly and helpful AI assistant. Respond warmly to the user's greeting or simple statement: {question}")
    return {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()


# --- Sidebar ---
with st.sidebar:
    st.subheader("Analyze Your Document")
    uploaded_file = st.file_uploader("Upload a PDF to begin analysis", type=['pdf'])

# --- Session State & Main Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I can answer questions about the pre-loaded Microsoft report, or you can upload a new document.")]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Determine which retriever to use and initialize the chain
retriever = get_retriever(uploaded_file)
if retriever:
    if st.session_state.rag_chain is None or uploaded_file:
        if uploaded_file and st.session_state.get("uploaded_file_name") != uploaded_file.name:
            st.session_state.messages = [AIMessage(content=f"Ready to chat about {uploaded_file.name}")]
            st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.rag_chain = get_rag_chain(retriever)
        st.session_state.conversational_chain = get_conversational_chain()
elif not os.path.exists(DEFAULT_INDEX_PATH):
     st.warning("Default knowledge base (`microsoft_faiss_index`) not found. Please run `rag_engine.py` first or upload a document.")

# --- Chat Interface ---
# Display messages from history
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)
        if hasattr(message, "metadata") and message.metadata.get("sources"):
            with st.expander(f"View Sources ({len(message.metadata['sources'])} chunks found)"):
                for i, source in enumerate(message.metadata["sources"]):
                    st.info(f"Source {i+1} (from Page {source.metadata.get('page', 'N/A')})")
                    st.code(source.page_content, language=None)
                    st.divider()

# Accept and process new user input
if prompt := st.chat_input("Ask a question...", disabled=(st.session_state.rag_chain is None)):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- THIS IS THE "SMART ROUTER" LOGIC ---
            # If the prompt is short, use the conversational chain. Otherwise, use the RAG chain.
            if len(prompt.split()) < 5:
                response = st.session_state.conversational_chain.invoke(prompt)
                st.markdown(response)
                # Add assistant response without sources
                st.session_state.messages.append(AIMessage(content=response))
            else:
                response_with_sources = st.session_state.rag_chain.invoke(prompt)
                answer = response_with_sources["answer"]
                sources = response_with_sources["context"]
                st.markdown(answer)
                # Add assistant response with sources
                st.session_state.messages.append(AIMessage(content=answer, metadata={"sources": sources}))
    st.rerun()