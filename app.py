import streamlit as st
import requests
import os

# --- الإعدادات الأساسية للـ API ---
# رابط السيرفر بتاعك على Render (تأكد إنه بدون / في الآخر)
API_BASE_URL = "https://financial-rag-api-d58u.onrender.com"
# الرقم السري الخفي لحماية الرفع (نفس اللي حطيناه في api.py)
UPLOAD_API_KEY = os.getenv("UPLOAD_API_KEY", "my_super_secret_key_123")

# --- إعدادات صفحة Streamlit ---
st.set_page_config(page_title="Financial Analyst AI", page_icon="📈", layout="wide")

# --- القائمة الجانبية (Sidebar) ---
with st.sidebar:
    st.header("📁 Data Ingestion")
    st.write("Upload a new Financial Report (PDF) to extract its Knowledge Graph and query it.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="200MB per file • PDF")
    
    if st.button("Process & Upload"):
        if uploaded_file is not None:
            # هنا بنعمل Request للسيرفر وبنبعت الـ API Key في الخفاء
            with st.spinner("Processing document... This might take a few minutes."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                headers = {"x-api-key": UPLOAD_API_KEY} # الباب الخلفي الآمن
                
                try:
                    response = requests.post(f"{API_BASE_URL}/upload", files=files, headers=headers)
                    if response.status_code == 200:
                        st.success("Document ingested and Knowledge Graph updated successfully!")
                    elif response.status_code == 401:
                        st.error("Unauthorized: Invalid API Key. Please check your configuration.")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: Could not reach the backend. {e}")
        else:
            st.warning("Please upload a file first.")
            
    st.markdown("---")
    st.markdown("🔧 **Powered by Neo4j & GPT-4o-Mini**")


# --- واجهة المحادثة الرئيسية (Main Chat Area) ---
st.title("📈 Financial Analyst AI (SOTA)")
st.markdown("*Enterprise-grade RAG supporting dynamic PDF ingestion.*")

# تهيئة الذاكرة الخاصة بالمحادثة
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your SOTA Financial Analyst. Upload a report from the sidebar or ask me a question!"}
    ]

# عرض المحادثات السابقة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# مربع إدخال الأسئلة
if prompt := st.chat_input("Ask about financial performance..."):
    # إضافة سؤال المستخدم للشاشة
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # تجهيز الرد من البوت
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # تجميع تاريخ المحادثة لإرساله للـ API
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            
            payload = {
                "query": prompt,
                "chat_history": history
            }
            
            # إرسال السؤال للـ API واستقبال الرد كتيار (Streaming)
            with requests.post(f"{API_BASE_URL}/chat", json=payload, stream=True) as r:
                if r.status_code == 200:
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            # عمل تأثير الكتابة الحية (Typing effect)
                            message_placeholder.markdown(full_response + "▌")
                    # عرض الرد النهائي بعد اكتمال الاستقبال
                    message_placeholder.markdown(full_response)
                else:
                    st.error(f"API Error {r.status_code}: Please ensure the backend is running.")
        except Exception as e:
            st.error(f"Connection error: {e}")
            
        # حفظ الرد النهائي في الذاكرة
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})