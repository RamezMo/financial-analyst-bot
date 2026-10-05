import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# تحميل المتغيرات البيئية
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_golden_dataset():
    pdf_path = "msft_report.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}. Please make sure the PDF is in the project folder.")
        return

    print("Loading PDF document for dataset generation...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # تقطيع المستند لمقاطع مناسبة
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print("Initializing Ragas models and embeddings...")
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0.7))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    # تهيئة TestsetGenerator بالمعاملات الصحيحة المعتمدة حديثاً
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embeddings
    )

    print("Generating 200 structured QA pairs (This will take a few minutes)...")
    
    # توليد العينة من مستندات لانغ تشين
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=200
    )

    # حفظ النتائج بصيغة JSON
    dataset_df = testset.to_pandas()
    output_file = "golden_dataset_200.json"
    dataset_df.to_json(output_file, orient="records", indent=4, force_ascii=False)
    
    print(f"Success! Generated 200 QA pairs and saved to '{output_file}'.")

if __name__ == "__main__":
    generate_golden_dataset()