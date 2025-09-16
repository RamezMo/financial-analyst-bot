# 🤖 Financial Analyst Q&A Bot

## 🚀 Live Demo

[**➡️ Insert Your Live Streamlit App Link Here After Deployment**]

---

## 📖 Project Overview

This project is an advanced, end-to-end RAG (Retrieval-Augmented Generation) application designed to act as a professional financial analyst. It allows users to have an intelligent, fact-based conversation with dense, lengthy financial documents, such as a company's annual 10-K report.

The application is built to be accurate, robust, and trustworthy, ensuring that all answers are based strictly on the content of the provided document. It showcases a full-cycle development process, from initial architectural design to prototyping, debugging complex AI behaviors, optimizing for performance and accuracy, and creating a polished user interface.

## ✨ Key Features

* **Document Q&A:** Users can ask complex questions in natural language and receive precise, fact-based answers sourced directly from the document.
* **On-the-Fly Document Analysis:** Features a user-friendly interface to upload any PDF document for instant analysis and conversation.
* **Pre-Processed Demo Mode:** Comes pre-loaded with Microsoft's 2024 Annual Report for an instant, high-performance demonstration of its capabilities.
* **"View Sources" Functionality:** For every answer generated, the application displays the exact text chunks from the source document that were used, providing complete transparency and building user trust.
* **High-Accuracy Embedding:** Utilizes a state-of-the-art OpenAI embedding model (`text-embedding-3-small`) to ensure a nuanced and accurate understanding of the document's complex financial content.
* **Intelligent "Smart Router":** The bot can differentiate between simple greetings and complex, document-related questions, allowing for a more natural and professional user interaction.
* **Robust & Safe:** The AI is explicitly instructed not to "hallucinate" or make up information. If an answer cannot be found in the text, the bot will state that, making it a reliable tool for factual analysis.

## 🛠️ Tech Stack

* **Language:** Python
* **Core Libraries:** Streamlit, LangChain, OpenAI
* **Document Loading:** `PyPDFLoader`
* **Vector Database:** `FAISS` (Facebook AI Similarity Search)
* **Embedding Model:** OpenAI `text-embedding-3-small`
* **LLM (Generator):** OpenAI `gpt-4o-mini`

## ⚙️ How It Works (RAG Architecture)

The application is built on a robust Retrieval-Augmented Generation (RAG) pipeline:

1.  **Load & Chunk:** The application ingests a PDF document and splits it into smaller, overlapping text chunks to fit within the language model's context window.
2.  **Embed & Store (The "Baking" Step):** Each chunk is converted into a numerical vector representation (an embedding) using OpenAI's powerful embedding model. These vectors are then stored in a high-speed, searchable FAISS vector index. This heavy processing is done once per document for maximum efficiency.
3.  **Retrieve:** When a user asks a question, their query is also converted into an embedding. The FAISS index is then searched to find the text chunks with the most similar semantic meaning to the question.
4.  **Generate:** The retrieved text chunks (the context) and the original question are passed to the `gpt-4o-mini` model with a precise prompt. The model then generates a final answer based only on the provided information.

## 🚀 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd financial-analyst-bot
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    * Create a file named `.env` in the root of the project.
    * Add your OpenAI API key to it:
        ```env
        OPENAI_API_KEY="sk-..."
        ```

5.  **(Optional but Recommended) Pre-process the default document:**
    To enable the instant demo mode, run the offline processing script once.
    ```bash
    python rag_engine.py
    ```

6.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```