import streamlit as st
import requests

st.set_page_config(page_title="Financial Analyst AI", page_icon="📈", layout="wide")

# القائمة الجانبية لرفع الملفات
with st.sidebar:
    st.header("📂 Data Ingestion")
    st.write("Upload a new Financial Report (PDF) to extract its Knowledge Graph and query it.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Process & Upload"):
        if uploaded_file is not None:
            with st.spinner("🧠 Analyzing Document & Building Knowledge Graph... This may take a few minutes."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post("http://localhost:8080/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                    else:
                        st.error(f"❌ Error processing document: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️️ Cannot connect to API. Is the server running on port 8080?")
        else:
            st.warning("Please select a file first.")
    
    st.markdown("---")
    st.write("🔧 **Powered by Neo4j & GPT-4o-Mini**")

# الشاشة الرئيسية للمحادثة
st.title("📈 Financial Analyst AI (SOTA)")
st.markdown("*Enterprise-grade RAG supporting dynamic PDF ingestion.*")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your SOTA Financial Analyst. Upload a report from the sidebar or ask me a question!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about financial performance..."):
    history_to_send = st.session_state.messages.copy()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            url = "http://localhost:8080/chat"
            payload = {
                "query": prompt, 
                "session_id": "default_session",
                "chat_history": history_to_send
            }
            headers = {"Content-Type": "application/json"}
            
            with requests.post(url, json=payload, headers=headers, stream=True) as response:
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    st.error(f"API Error: {response.status_code}")
                    
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Connection Error: Please make sure the FastAPI server (api.py) is running on port 8080.")
            
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})