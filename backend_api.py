from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document
import glob
from dotenv import load_dotenv
import logging

# Production configuration
if os.environ.get('RENDER'):
    # Running on Render
    os.makedirs("/tmp/documents", exist_ok=True)
    os.makedirs("/tmp/chroma_db", exist_ok=True)


# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for QA chain
qa_chain = None
vectorstore = None

def setup_document_samples():
    """Create sample documents for the assistant"""
    # Use /tmp for temporary storage on Render
    doc_path = "/tmp/documents" if os.environ.get('RENDER') else "documents"
    os.makedirs(doc_path, exist_ok=True)
    
    # Create sample return policy
    with open(f"{doc_path}/return_policy.txt", "w") as f:
        f.write("""
Walmart Return Policy

1. Most items can be returned within 90 days of purchase
2. Electronics must be returned within 30 days
3. Items must be in original packaging with all accessories
4. Receipt or order number required for returns
5. Refunds issued to original payment method within 5-7 business days

Exclusions:
- Opened software, movies, and music
- Gift cards
- Personalized items
- Perishable goods
""")
       
    with open(f"{doc_path}/return_policy.txt", "w") as f:
        f.write("""
Walmart Points System

- Purchasing: Earn 5 percent of product MRP in points
- Recycling: Earn 50 percent of current product value in points
- Returns: Deduct 1.5x points earned originally
- Tier Boosts: Silver/Gold tiers get higher earnings or perks
- Transfer Rules: Specific conversion rates with partner brands
""")
        
    # Create sample shipping policy
    with open(f"{doc_path}/return_policy.txt", "w") as f:
        f.write("""
Walmart Shipping Options

Standard Shipping:
- Free on orders over $35
- 3-5 business days
- Available for most items

Express Shipping:
- $5.99 flat rate
- 2 business days
- Available for eligible items

NextDay Delivery:
- $8.99 flat rate
- Next business day delivery by 9pm
- Available in select areas

International Shipping:
- Available to over 200 countries
- Shipping costs vary by destination
- Customs fees may apply
""")
    
    # Create sample product catalog
    with open(f"{doc_path}/return_policy.txt", "w") as f:
        f.write("""
Walmart Electronics Catalog

1. Apple iPhone 15 Pro
   - Price: $999
   - Colors: Space Black, Silver, Gold, Blue
   - Features: A17 Pro chip, 6.1" Super Retina XDR display, 48MP camera

2. Samsung 65" QLED 4K Smart TV
   - Price: $799
   - Features: Quantum HDR, Object Tracking Sound, Smart Hub with streaming apps

3. Sony WH-1000XM5 Wireless Headphones
   - Price: $349
   - Features: Industry-leading noise cancellation, 30-hour battery, multipoint connection

4. Ninja Foodi 10-in-1 Air Fry Oven
   - Price: $199
   - Features: Air fry, roast, bake, toast, dehydrate functions, 1800W power
""")
    
    with open(f"{doc_path}/return_policy.txt", "w") as f:
        f.write("""
Walmart Store Hours

Regular Hours:
- Monday-Sunday: 6:00 AM - 11:00 PM
- Pharmacy: 9:00 AM - 8:00 PM (Mon-Fri), 9:00 AM - 6:00 PM (Sat-Sun)
- Vision Center: 9:00 AM - 8:00 PM (Mon-Fri), 9:00 AM - 6:00 PM (Sat)

Holiday Hours:
- Thanksgiving: Closed
- Christmas: Closed
- New Year's Day: 10:00 AM - 8:00 PM
- Other holidays may have modified hours

Services:
- Grocery Pickup: Available 8:00 AM - 8:00 PM
- Delivery: Available in select areas
- Auto Care Center: 7:00 AM - 7:00 PM
""")
    
    logger.info("Sample documents created successfully")
    return True

def load_documents():
    """Load and process documents from a directory"""
    documents = []
    
    # Load PDF files
    for pdf_file in glob.glob("documents/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        documents.extend(pages)
    
    # Load text files
    for txt_file in glob.glob("documents/*.txt"):
        loader = TextLoader(txt_file)
        pages = loader.load()
        documents.extend(pages)
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents

def initialize_qa_system():
    """Initialize the QA system with documents and embeddings"""
    global qa_chain, vectorstore
    
    try:
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        # Create sample documents if they don't exist
        if not os.path.exists("documents") or not os.listdir("documents"):
            logger.info("Creating sample documents...")
            setup_document_samples()
        
        # Load and process documents
        documents = load_documents()
        
        if not documents:
            logger.warning("No documents found. Creating fallback content.")
            documents = [
                Document(page_content="Walmart Return Policy: 90 days for most items", metadata={"source": "fallback.txt"}),
                Document(page_content="Walmart Shipping: Free on orders over $35", metadata={"source": "fallback.txt"})
            ]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        # Create embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory="./chroma_db"
        )
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are Kendall, a helpful and friendly Walmart customer service assistant. Use the provided context to answer the customer's question accurately and helpfully.

Guidelines:
- Be conversational and friendly
- Provide specific information when available
- If you don't know something, say so and suggest contacting customer support
- Always be helpful and solution-oriented
- Use the context provided to give accurate answers

Context:
{context}

Customer Question:
{question}

Your helpful response:
""",
        )
        
        # Set up Groq LLM
        llm = ChatGroq(
            temperature=0.2,
            model_name="llama3-70b-8192",
            api_key=groq_api_key
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        logger.info("QA system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing QA system: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Walmart Chatbot API is running"})

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        if not qa_chain:
            return jsonify({"error": "QA system not initialized"}), 500
        
        # Get answer from QA system
        result = qa_chain({"query": question})
        answer = result["result"]
        
        # Extract sources
        sources = []
        if "source_documents" in result:
            sources = list(set([
                os.path.basename(doc.metadata.get('source', 'Unknown'))
                for doc in result["source_documents"]
            ]))
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "Sorry, I'm having trouble processing your request right now.",
            "status": "error"
        }), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    """Reset chat history endpoint"""
    return jsonify({"message": "Chat reset successfully", "status": "success"})

# if __name__ == '__main__':
#     # Initialize QA system on startup
#     logger.info("Starting Walmart Chatbot API...")
#     if initialize_qa_system():
#         logger.info("QA system ready!")
#         app.run(debug=True, host='0.0.0.0', port=8000)
#     else:
#         logger.error("Failed to initialize QA system. Exiting.")
#         exit(1)

if __name__ == '__main__':
    # Initialize QA system on startup
    logger.info("Starting Walmart Chatbot API...")
    if initialize_qa_system():
        logger.info("QA system ready!")
        # For production, use the port from environment
        port = int(os.environ.get('PORT', 8000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        logger.error("Failed to initialize QA system. Exiting.")
        exit(1)