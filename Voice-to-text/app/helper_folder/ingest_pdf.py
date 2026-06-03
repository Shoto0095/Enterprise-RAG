from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ..logger import get_logger

logger = get_logger("ingest_pdf")
# Initialize embeddings once at module level
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest_pdf(pdf_path, collection_name="project_kb", db_path="./chroma_db"):
    """
    Ingest PDF into Chroma vector database
    
    Args:
        pdf_path: Path to the PDF file
        collection_name: Name of the Chroma collection
        db_path: Path to the Chroma database
        
    Returns:
        bool: True if successful
    """
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=db_path)
        db.add_documents(chunks)

        logger.info(f" PDF '{pdf_path}' embedded & stored!")
        return True
    except Exception as e:
        logger.error(f"Error ingesting PDF: {str(e)}")
        return False

# if __name__ == "__main__":
#     pdf_path = input("Enter PDF path: ")
#     ingest_pdf(pdf_path)
