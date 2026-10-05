import os
import pymupdf4llm
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector

# تحميل المتغيرات البيئية
load_dotenv(override=True)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def run_advanced_ingestion():
    print("1. Connecting to Neo4j & Formatting Database...")
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    graph.query("MATCH (n) DETACH DELETE n")
    try:
        graph.query("DROP INDEX financial_bot_index IF EXISTS")
    except Exception:
        pass

    print("2. 📄 Advanced PDF Parsing (Extracting Tables as Markdown)...")
    # هذه الخطوة السحرية تحول الـ PDF بجدواله إلى Markdown مهيكل تماماً
    md_text = pymupdf4llm.to_markdown("msft_report.pdf")
    
    # تقسيم النص بناءً على عناوين الـ Markdown للحفاظ على ترابط أقسام التقرير
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = md_splitter.split_text(md_text)

    print("3. 🧠 Semantic Chunking on Markdown Sections...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    semantic_chunker = SemanticChunker(embeddings=embeddings)
    
    # تطبيق التقطيع الدلالي فوق التقطيع الهيكلي لضمان أقصى دقة
    final_chunks = semantic_chunker.split_documents(md_splits)
    print(f"   ✅ Created {len(final_chunks)} Advanced Markdown/Semantic Chunks.")

    print("4. 🧱 Storing Vector Documents (Vector DB)...")
    Neo4jVector.from_documents(
        final_chunks,
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        index_name="financial_bot_index"
    )
    print("   ✅ Vector Store is fully populated with Structured Markdown.")

    print("5. 🕸️ Extracting Knowledge Graph (Graph DB)...")
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    allowed_nodes = ["Company", "FinancialMetric", "Product", "Value", "Year"]
    allowed_relationships = ["REPORTED_REVENUE", "INCREASED_BY", "DECREASED_BY", "PRODUCES", "HAS_VALUE"]
    
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships
    )
    
    graph_documents = []
    for i, chunk in enumerate(final_chunks):
        try:
            if (i+1) % 20 == 0:
                print(f"   Processed {i+1}/{len(final_chunks)} chunks for graph extraction...")
            extracted = transformer.convert_to_graph_documents([chunk])
            graph_documents.extend(extracted)
        except Exception:
            pass

    print(f"6. Saving {len(graph_documents)} Graph Relationships to Neo4j...")
    graph.add_graph_documents(
        graph_documents, 
        baseEntityLabel=True, 
        include_source=True
    )
    
    print("\n✅ Success! Advanced SOTA Database Ingestion Complete.")

if __name__ == "__main__":
    run_advanced_ingestion()