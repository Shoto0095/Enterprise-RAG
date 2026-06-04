"""
RAG orchestration: retriever, prompt and chain.

Expose `invoke(query)` and `restart_chatbot()` for callers.
"""

from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler
from .llm import GeminiLLM
from .config import settings

# from .embedding_wrapper import CachedEmbeddingFunction
# cached_embeddings = CachedEmbeddingFunction()
cached_embeddings = None  # <-- keep as-is if already defined elsewhere

# -----------------------------
# Internal mutable state
# -----------------------------
_db = None
_retriever = None
_rag_chain = None

# -----------------------------
# Prompt template
# -----------------------------
template = """
You are an AI assistant that answers user questions strictly based on the provided context.

The context is generated from uploaded video content and PDFs and may contain informal language, filler words, or minor transcription errors.

Instructions:

- Use ONLY the information explicitly present in the provided context.
- Based on the context and question answer briefly and correctly so that user can understand it clearly and without any problems.
- Do not say exact words from the content, but rephrase them in your own words.
- If the context does not contain the answer, think if you can infer it from the context. If you can infer it, provide the inferred answer clearly indicating it is inferred if not say "I don’t have the information about that yet."
- Do NOT assume, infer, or add any information that is not clearly stated in the context.
- Always remember the PREVIOUS MESSAGES in the conversation to maintain context.
- User can have a casual conversation also. Respond accordingly do not look for context in database for that act like normal chatbot.
- If the question is a greeting (e.g., hello, hi, hey, bye), respond naturally and briefly.
- Avoid hallucinations
- User can ask questions based on the previous conversations also, so make sure to answer them too based on the context.
- Maintain a clear, professional, and neutral tone.
- Format the final answer in valid HTML.
- Do NOT include <html> or <body> tags. Use only child tags such as <p>, <ul>, <li>, <strong>, etc.

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

# -----------------------------
# Internal initializer
# -----------------------------
def _init_rag():
    """
    Initialize or reinitialize RAG components.
    This DOES NOT delete the DB.
    It only reloads latest data from disk.
    """
    global _db, _retriever, _rag_chain

    _db = Chroma(
        persist_directory=settings.CHROMA_DIR,
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=cached_embeddings,
    )

    _retriever = _db.as_retriever(search_kwargs={"k": 3})

    llm = GeminiLLM(
        model=settings.GEMINI_MODEL,
        streaming=False,
        callbacks=[StdOutCallbackHandler()],
    )

    _rag_chain = (
        {"context": _retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

# -----------------------------
# Public API
# -----------------------------
def restart_chatbot():
    """
    Refresh chatbot memory and retriever.
    - Keeps all existing DB data
    - Picks up newly ingested documents
    - Clears in-memory chain
    """
    global _db, _retriever, _rag_chain
    _db = None
    _retriever = None
    _rag_chain = None
    _init_rag()


def invoke(query: str, session_id: str = None):
    """
    Invoke the RAG chain synchronously.
    """
    if _rag_chain is None:
        _init_rag()
    config = {"configurable": {"thread_id": session_id}} if session_id else None
    return _rag_chain.invoke(query, config=config)


def invoke_stream(query: str, session_id: str = None):
    """
    Invoke the RAG chain with streaming output.
    Yields chunks of the response as they are generated.
    """
    if _rag_chain is None:
        _init_rag()
    
    config = {"configurable": {"thread_id": session_id}} if session_id else None
    
    # Get context from retriever
    context_docs = _retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in context_docs])
    
    # Format prompt
    formatted_prompt = prompt.format(context=context, question=query)
    
    # Get LLM instance and stream
    llm = GeminiLLM(
        model=settings.GEMINI_MODEL,
        streaming=True,
    )
    
    # Stream from LLM
    for chunk in llm.stream(formatted_prompt):
        yield chunk
