import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_experimental.graph_transformers import LLMGraphTransformer


# إعداد التسجيل (Logging) لمنع التجاهل الصامت للأخطاء
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FinancialRAGEngine")

load_dotenv(override=True)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

class FinancialRAGEngine:
    def __init__(self):
        logger.info("Initializing Production Financial RAG Engine...")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
        
        # التعديل هنا: إضافة refresh_schema=False
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            refresh_schema=False
        )
        
        self._ensure_indexes()
        
        self.vector_store = Neo4jVector.from_existing_index(
            embedding=self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            index_name="financial_bot_index",
            keyword_index_name="keyword",
            search_type="hybrid"
        )
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # 1. طبقة إعادة الصياغة
        rewrite_prompt = PromptTemplate.from_template(
            "You are a financial search assistant. Rewrite the following user query to be highly descriptive for a database search. "
            "Include common financial synonyms. Keep it concise. Original Query: {query}"
        )
        self.query_rewriter = rewrite_prompt | self.llm | StrOutputParser()
        
        # 2. طبقة استخراج الكيانات
        entity_prompt = PromptTemplate.from_template(
            "Extract key financial entities from the query. "
            "ALSO include their formal financial synonyms or likely product names. "
            "Return ONLY a comma-separated list of these exact terms. Query: {query}"
        )
        self.entity_extractor = entity_prompt | self.llm | StrOutputParser()

        # 3. الـ Prompt المعتمد (مزود بفهم شامل للهجة المصرية والذاكرة)
        self.final_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly professional Financial Analyst AI.

INSTRUCTIONS:
1. CONVERSATION & MEMORY: You have a persistent memory of the chat. If the user introduces themselves, asks about their name, or makes casual conversation, warmly respond using the Chat History.
* DIALECT NOTE: The user speaks Egyptian Arabic. Understand terms naturally (e.g., "اية" or "ايه" means "What", not the name "Aya" unless explicitly stated). Treat all casual Egyptian conversational phrases as Small Talk.
IGNORE the retrieved financial context for these conversational questions.
2. FINANCIAL QUERIES: For analytical or factual questions, you MUST answer using ONLY the provided Retrieved Context. If the answer is not in the context, clearly state that you do not have the information. NEVER hallucinate.
3. Prioritize exact numbers from GRAPH FACTS or tables in the TEXT.

Retrieved Context:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])

    def _ensure_indexes(self):
        try:
            self.graph.query(
                "CREATE FULLTEXT INDEX entity_id_index IF NOT EXISTS FOR (n:__Entity__) ON EACH [n.id]"
            )
            logger.info("Entity full-text index verified/created.")
        except Exception as e:
            logger.warning(f"Unable to create full-text index on entities: {e}")

    async def _fetch_graph_facts(self, query: str) -> list[str]:
        graph_facts = []
        try:
            entities_str = await self.entity_extractor.ainvoke({"query": query})
            entities = [e.strip() for e in entities_str.split(',') if e.strip()]
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            return graph_facts

        for entity in entities:
            sanitized_entity = "".join(c for c in entity if c.isalnum() or c.isspace()).strip()
            if not sanitized_entity:
                continue

            records = []
            try:
                index_cypher = """
                CALL db.index.fulltext.queryNodes("entity_id_index", $search_term) YIELD node
                MATCH (node)-[r]-(m)
                RETURN node.id AS source, type(r) AS rel, m.id AS target
                LIMIT 10
                """
                records = self.graph.query(index_cypher, params={"search_term": f"*{sanitized_entity}*"})
            except Exception as e:
                logger.debug(f"Full-text query not applicable for '{entity}': {e}")

            if not records:
                try:
                    fallback_cypher = """
                    MATCH (n)-[r]-(m)
                    WHERE toLower(n.id) CONTAINS toLower($entity) OR toLower(m.id) CONTAINS toLower($entity)
                    RETURN n.id AS source, type(r) AS rel, m.id AS target
                    LIMIT 10
                    """
                    records = self.graph.query(fallback_cypher, params={"entity": sanitized_entity})
                except Exception as e:
                    logger.error(f"Fallback graph query failed for '{entity}': {e}", exc_info=True)

            for record in records:
                graph_facts.append(f"{record['source']} -> [{record['rel']}] -> {record['target']}")

        return list(set(graph_facts))

    async def process_query_stream(self, user_query: str, chat_history: list = None):
        # تحويل الذاكرة إلى كائنات تدعمها LangChain (HumanMessage / AIMessage)
        formatted_history = []
        if chat_history:
            for msg in chat_history[-6:]:  # الاحتفاظ بآخر 6 رسائل فقط
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(AIMessage(content=msg["content"]))

        # الاسترجاع
        try:
            optimized_query = await self.query_rewriter.ainvoke({"query": user_query})
        except Exception as e:
            logger.warning(f"Query rewriter failed: {e}")
            optimized_query = user_query

        try:
            docs = await self.vector_retriever.ainvoke(optimized_query)
            vector_context = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Vector retrieval error: {e}", exc_info=True)
            vector_context = "No document text retrieved."

        graph_facts = await self._fetch_graph_facts(optimized_query)
        graph_context_str = "\n".join(graph_facts) if graph_facts else "No exact graph facts found."
        
        combined_context = f"--- GRAPH FACTS ---\n{graph_context_str}\n\n--- DOCUMENT TEXT ---\n{vector_context}"

        # التنفيذ مع الذاكرة الهيكلية
        chain = self.final_prompt | self.llm | StrOutputParser()
        return chain.astream({
            "context": combined_context,
            "chat_history": formatted_history,
            "query": user_query
        })


    async def ingest_document(self, file_path: str):
        """دالة ديناميكية لهضم ملفات الـ PDF الجديدة ورفعها على قاعدة البيانات"""
        logger.info(f"Starting ingestion process for {file_path}")
        
        # 1. تحويل PDF إلى Markdown
        md_text = pymupdf4llm.to_markdown(file_path)
        
        # 2. التقطيع الهيكلي (بناءً على العناوين)
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_docs = markdown_splitter.split_text(md_text)
        
        # 3. التقطيع الدلالي (Semantic Chunking)
        semantic_chunker = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
        final_docs = semantic_chunker.transform_documents(md_docs)
        logger.info(f"Generated {len(final_docs)} semantic chunks.")

        # 4. رفع النصوص كمتجهات (Vector Store)
        Neo4jVector.from_documents(
            final_docs,
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            index_name="financial_bot_index",
            keyword_index_name="keyword",
            search_type="hybrid"
        )
        
        # 5. استخراج الجراف المعرفي (Knowledge Graph)
        llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["Company", "Product", "FinancialMetric", "Person", "Market", "Event"],
            allowed_relationships=["HAS_REVENUE", "ACQUIRED", "COMPETES_WITH", "PRODUCES", "LED_BY", "REPORTED"]
        )
        
        # هذه الخطوة تستغرق وقتاً لأنها تتصل بـ LLM
        graph_documents = llm_transformer.convert_to_graph_documents(final_docs)
        self.graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
        
        # تحديث الفهارس النصية
        self._ensure_indexes()
        logger.info("Ingestion completed successfully!")