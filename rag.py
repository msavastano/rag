import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from environment variables
# Make sure to set your GOOGLE_API_KEY in a .env file in the root of your project
# GOOGLE_API_KEY="YOUR_API_KEY"
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Please set the GOOGLE_API_KEY environment variable.")
else:
    # 1. Load the document
    loader = TextLoader('gemini_info.txt')
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # 3. Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # 4. Create a FAISS vector store
    db = FAISS.from_documents(docs, embeddings)

    # 5. Create a RetrievalQA chain
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")

    # 6. Run a sample query
    query = "What are the different sizes of Gemini models?"
    retrieved_docs = db.similarity_search(query)
    answer = chain.run(input_documents=retrieved_docs, question=query)

    print(f"Query: {query}")
    print(f"Answer: {answer}")
