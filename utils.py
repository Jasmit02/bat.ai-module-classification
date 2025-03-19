"""
Utility functions for the Module Classification System.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import HypotheticalDocumentEmbedder

from config import config

class LineListOutputParser(StrOutputParser):
    """Parse the output of an LLM call to a list of strings."""
    def parse(self, text: str) -> List[str]:
        """Parse the output text into a list of lines."""
        return [line.strip() for line in text.strip().split("\n") if line.strip()]

def initialize_clients() -> Tuple[ChatNVIDIA, NVIDIAEmbeddings, RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter, InMemoryStore]:
    """
    Initialize the necessary clients and resources for the application.
    
    Returns:
        tuple: Contains the following elements:
            - chat_client: ChatNVIDIA instance for language model interactions
            - embed_client: NVIDIAEmbeddings instance for text embedding
            - parent_splitter: Text splitter for parent documents
            - child_splitter: Text splitter for child documents
            - store: InMemoryStore for document storage
    """
    chat_client = ChatNVIDIA(
        model=config.default_chat_model,
        api_key=config.nvidia_api_key,
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096
    )
    
    embed_client = NVIDIAEmbeddings(
        model=config.embed_model,
        api_key=config.nvidia_api_key,
        truncate="NONE"
    )
    
    # Initialize text splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config.parent_chunk_size)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=config.child_chunk_size)
    
    # Initialize document store
    store = InMemoryStore()
    
    return chat_client, embed_client, parent_splitter, child_splitter, store

def load_vector_store(embed_client: NVIDIAEmbeddings) -> FAISS:
    """
    Load the FAISS vector store.
    
    Args:
        embed_client: NVIDIAEmbeddings instance
        
    Returns:
        FAISS: Loaded vector store
    """
    return FAISS.load_local(
        config.index_path,
        embeddings=embed_client,
        allow_dangerous_deserialization=True
    )

def setup_retrievers(vector_store: FAISS, store: InMemoryStore, 
                    parent_splitter: RecursiveCharacterTextSplitter, 
                    child_splitter: RecursiveCharacterTextSplitter,
                    chat_client: ChatNVIDIA,
                    embed_client: NVIDIAEmbeddings) -> Any:
    """
    Set up retrievers based on configuration.
    
    Args:
        vector_store: FAISS vector store
        store: InMemoryStore for document storage
        parent_splitter: Text splitter for parent documents
        child_splitter: Text splitter for child documents
        chat_client: ChatNVIDIA instance
        embed_client: NVIDIAEmbeddings instance
        
    Returns:
        Any: Configured retriever
    """
    # Initialize semantic retriever
    semantic_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": config.score_threshold,
            "k": config.k_value
        }
    )
    
    # Initialize ParentDocumentRetriever
    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    # Create hybrid ensemble retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[parent_document_retriever, semantic_retriever],
        weights=[0.5, 0.5]
    )
    
    # Select base retriever based on configuration
        
    # Apply HyDE if configured
    
    active_retriever = hybrid_retriever
    
    # Apply MultiQueryRetriever if configured
    if config.use_multi_query:
        # Clear any existing handlers to avoid duplicates
        logger = logging.getLogger("langchain.retrievers.multi_query")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create custom prompt for generating multiple queries
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are a world class QA expert who works on bug classification. Generate multiple versions of the query to improve retrieval making sure to cover all the aspects of the query and keep the meaning and format intact. It can be also in a format like Project Name, Project Description, Error Name, Error Description, etc. Your task is to generate {num_queries} 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
        
        # Create the multiquery retriever with the custom prompt
        output_parser = LineListOutputParser()
        llm_chain = query_prompt.partial(num_queries=str(config.num_alt_queries)) | chat_client | output_parser
        
        multi_query_retriever = MultiQueryRetriever(
            retriever=active_retriever,
            llm_chain=llm_chain,
            parser_key="lines"
        )
        print("MultiQueryRetriever initialized successfully")
        return multi_query_retriever
    
    return active_retriever