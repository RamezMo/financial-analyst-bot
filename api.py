from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Optional
import shutil
import os
from engine import FinancialRAGEngine

app = FastAPI(title="Financial Analyst RAG API", version="1.0")

# إضافة CORS عشان الواجهة تقدر تكلم السيرفر بدون قيود أمنية من المتصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يسمح لأي واجهة (زي Streamlit) بالاتصال
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = FinancialRAGEngine()

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"
    chat_history: Optional[List[Dict[str, str]]] = []

# إضافة الـ Health Check لـ UptimeRobot عشان يفضل السيرفر شغال (حل مشكلة 404)
@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Financial Analyst RAG API is running!"}

@app.post("/chat")
async def chat_stream(request: QueryRequest):
    user_query = request.query
    chat_history = request.chat_history
    
    generator = await rag_engine.process_query_stream(user_query, chat_history)
    
    async def event_generator():
        try:
            async for chunk in generator:
                yield chunk
        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"\n\n⚠️ عذراً، حدث خطأ: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_file_path = f"temp_{file.filename}"
    try:
        # حفظ الملف مؤقتاً
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # تمرير الملف للمحرك لهضمه
        await rag_engine.ingest_document(temp_file_path)
        
        return {"message": "Document ingested and Knowledge Graph updated successfully!"}
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # مسح الملف المؤقت بعد الانتهاء من المعالجة
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    # التعديل الاحترافي: قراءة البورت من بيئة التشغيل (Render) أو استخدام 8080 افتراضياً
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting FastAPI Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)