import sys
from unittest.mock import MagicMock

# --- Monkey Patch ---
import langchain_community
sys.modules['langchain_community.chat_models'] = MagicMock()
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.chat_models.vertexai'].ChatVertexAI = MagicMock
try:
    import langchain_community.llms
except ImportError:
    sys.modules['langchain_community.llms'] = MagicMock()
sys.modules['langchain_community.llms'].VertexAI = MagicMock
# --------------------

import os
import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def run_real_world_evaluation():
    print("1. Loading FULL Golden Dataset (200 Questions)...")
    with open("golden_dataset_200.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # إزالة التقطيع لاختبار الـ 200 سؤال كاملة
    questions = [item["user_input"] for item in data]  
    ground_truths = [item["reference"] for item in data]

    print("2. Initializing SOTA Hybrid Retrievers with Anti-Leakage Architectures...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

    try:
        graph.query("CREATE FULLTEXT INDEX keyword IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]")
    except Exception:
        pass 

    vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE,
        index_name="financial_bot_index", keyword_index_name="keyword", search_type="hybrid" 
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # --- الحل 1: Query Rewriter (لمنع تسريب المفردات) ---
    rewrite_prompt = PromptTemplate.from_template(
        "You are a financial search assistant. Rewrite the following user query to be highly descriptive for a database search. "
        "Include common financial synonyms (e.g., if they say 'sales', add 'revenue'). "
        "Keep it concise but comprehensive. Original Query: {query}"
    )
    query_rewriter_chain = rewrite_prompt | llm | StrOutputParser()

    # --- الحل 2: Fuzzy Entity Resolver (لمنع جمود الجراف) ---
    entity_extractor_prompt = PromptTemplate.from_template(
        "Extract key financial entities from the query. "
        "ALSO include their formal financial synonyms or likely product names (e.g., 'Cloud' -> 'Azure, Cloud Services'). "
        "Return ONLY a comma-separated list of these exact terms and synonyms. Query: {query}"
    )
    entity_chain = entity_extractor_prompt | llm | StrOutputParser()

    print("3. Executing Real-World Pipeline on 200 queries (This will take time)...")
    answers = []
    contexts = []

    for i, query in enumerate(questions):
        if (i + 1) % 10 == 0:
            print(f"   Processing query {i + 1}/200...")

        # أ. إعادة كتابة السؤال لمعالجة الكلمات العامية
        try:
            optimized_query = query_rewriter_chain.invoke({"query": query})
        except Exception:
            optimized_query = query
            
        # ب. البحث بالفيكتور باستخدام السؤال المُحسن
        docs = vector_retriever.invoke(optimized_query)
        vector_context = "\n".join([doc.page_content for doc in docs])
        
        # ج. استخراج الكيانات الموسعة (Fuzzy) والبحث في الجراف
        graph_facts = []
        try:
            entities_str = entity_chain.invoke({"query": optimized_query})
            entities = [e.strip() for e in entities_str.split(',') if e.strip()]
            
            for entity in entities:
                cypher_query = """
                MATCH (n)-[r]-(m) 
                WHERE toLower(n.id) CONTAINS toLower($entity) OR toLower(m.id) CONTAINS toLower($entity)
                RETURN n.id AS source, type(r) AS rel, m.id AS target LIMIT 10
                """
                res = graph.query(cypher_query, params={"entity": entity})
                for record in res:
                    graph_facts.append(f"{record['source']} -> [{record['rel']}] -> {record['target']}")
        except Exception:
            pass
        
        graph_facts = list(set(graph_facts))
        graph_context_str = "\n".join(graph_facts) if graph_facts else "No exact graph facts found."

        # د. دمج السياق وتوليد الإجابة
        combined_context = f"--- GRAPH FACTS ---\n{graph_context_str}\n\n--- DOCUMENT TEXT ---\n{vector_context}"
        contexts.append([combined_context])
        
        final_prompt = f"""You are a precise Financial Analyst. Answer the question using ONLY the provided context.
Context includes both structured GRAPH FACTS and Markdown DOCUMENT TEXT. 
Prioritize exact numbers from GRAPH FACTS or tables in the TEXT.

Context:
{combined_context}

User Question: {query}
Answer:"""

        response = llm.invoke(final_prompt)
        answers.append(response.content)

    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    eval_dataset = Dataset.from_dict(eval_data)

    print("4. Running RAGAS Evaluation on 200 items...")
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    eval_embeddings = LangchainEmbeddingsWrapper(embeddings)

    result = evaluate(
        eval_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=eval_llm, embeddings=eval_embeddings
    )

    print("\n=== REAL-WORLD SOTA EVALUATION RESULTS (200 QA) ===")
    print(result)
    result.to_pandas().to_csv("evaluation_real_world_report.csv", index=False)
    print("\nReport saved to 'evaluation_real_world_report.csv'")

if __name__ == "__main__":
    run_real_world_evaluation()