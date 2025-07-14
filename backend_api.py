from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import glob
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Setup Flask and CORS
app = Flask(__name__)
CORS(app)

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
qa_chain = None
vectorstore = None

# Determine doc path based on environment
DOC_PATH = "/tmp/documents" if os.environ.get("RENDER") else "documents"
DB_PATH = "/tmp/chroma_db" if os.environ.get("RENDER") else "./chroma_db"
os.makedirs(DOC_PATH, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)


def create_sample_documents():
    """Generate static example documents in DOC_PATH"""
    docs = {
        "return_policy.txt": """
Walmart Return Policy
1. Most items: return within 90 days
2. Electronics: return within 30 days
3. Must have original packaging and receipt
4. Refund in 5–7 business days
""",
        "points_system.txt": """
Walmart Points System
- Purchases: Earn 5% of MRP
- Recycling: Earn 50% of product value
- Returns: Deduct 1.5x earned points
- Tiers: Silver/Gold for perks
""",
        "shipping.txt": """
Shipping Options
- Free over $35: 3–5 days
- Express: $5.99, 2-day
- NextDay: $8.99, by 9PM
- International: Custom rates
""",
        "store_hours.txt": """
Store Hours
- Open: 6AM–11PM daily
- Pharmacy: 9AM–8PM
- Holidays: Closed on major holidays
""",
    }
    for filename, content in docs.items():
        with open(os.path.join(DOC_PATH, filename), "w") as f:
            f.write(content)
    logger.info("Sample documents created.")


def load_documents():
    """Load .pdf and .txt files from DOC_PATH"""
    documents = []
    for txt_file in glob.glob(f"{DOC_PATH}/*.txt"):
        loader = TextLoader(txt_file)
        documents.extend(loader.load())
    for pdf_file in glob.glob(f"{DOC_PATH}/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    return documents


def initialize_qa():
    """Initializes vectorstore and QA chain using LangChain & Groq"""
    global qa_chain, vectorstore
    try:
        # Validate Groq key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY")

        # Ensure documents exist
        if not os.listdir(DOC_PATH):
            logger.info("No documents found. Creating samples...")
            create_sample_documents()

        # Load & split documents
        raw_docs = load_documents()
        if not raw_docs:
            raw_docs = [Document(page_content="Fallback: Walmart Policy", metadata={"source": "default"})]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(raw_docs)

        # Embed with HuggingFace
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Store embeddings
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_PATH
        )

        # Prompt template
        template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are Kendall, a friendly Walmart assistant. Use only the provided context to answer the customer's question.

Context:
{context}

Question:
{question}

Answer as Kendall:
"""
        )

        # LLM and QA chain
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.2,
            api_key=groq_api_key
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": template},
            return_source_documents=True
        )

        logger.info("QA system initialized.")
        return True

    except Exception as e:
        logger.exception(f"Failed to initialize QA system: {str(e)}")
        return False


# Initialize QA on import (works with gunicorn)
initialize_qa()


# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Walmart Chatbot API is up"})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Please enter a question.", "sources": [], "status": "error"}), 400

        if not qa_chain:
            return jsonify({"answer": "Kendall is still warming up. Try again shortly.", "sources": [], "status": "error"}), 503

        result = qa_chain({"query": question})
        answer = result.get("result", "I'm not sure. Please rephrase.").strip()

        sources = list(set(
            os.path.basename(doc.metadata.get('source', 'Unknown'))
            for doc in result.get("source_documents", [])
        ))

        return jsonify({"answer": answer, "sources": sources, "status": "success"})

    except Exception as e:
        logger.exception("Chat error")
        return jsonify({
            "answer": "An error occurred. Please try again later.",
            "sources": [],
            "status": "error"
        }), 500


@app.route('/reset', methods=['POST'])
def reset_chat():
    return jsonify({"message": "Chat reset complete", "status": "success"})


# Optional: Local development server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Running on http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
