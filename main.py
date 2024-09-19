from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.odt import UnstructuredODTLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
import os
from uuid import uuid4
import argparse
import pyperclip
import logging
import time
import PyPDF2

SUPPORTED_DOC_TYPES = ["pdf", "md", "html", "txt", "odt", "docx", "doc", "pptx", "ppt"]

# Initialize embedding and vector store
def initialize_embeddings(model_name):
    model_kwargs = {'device': 'cpu',  'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def build_query(query):
    # Prepare a prompt given an instruction
    # instruction = 'Given a web search query in French, retrieve relevant passages that answer the query.'
    # prompt = f'<instruct>{instruction}\n<query>{query}'

    return query

def initialize_vector_store(embeddings, db_path="./chroma_langchain_db"):
    return Chroma(
        collection_name="articles",
        embedding_function=embeddings,
        persist_directory=db_path,  # Persistent storage for local data
    )

# Load and process documents from PDFs
def load_and_process_files(pdf_folder, vector_store, logger, batch_size=5, doc_types="pdf"):
    doc_types = doc_types.split("/")

    # Check if any document type is not in the supported list
    for doc_type in doc_types:
        if doc_type not in SUPPORTED_DOC_TYPES:
            raise ValueError(f"Unsupported document type: '{doc_type}'")
    
    collection = vector_store.get(limit=2)

    # If the collection is empty, create a new one
    if len(collection['ids']) > 0:
        return

    files = sorted([os.path.join(root, f) 
                for root, dirs, files in os.walk(pdf_folder) 
                for f in files if f.endswith(tuple([f".{ext}" for ext in doc_types]))])
    total = len(files)
    documents = []

    logger.info(f"Total discovered PDF : {total}")
    
    error_count = 0
    for idx, file in enumerate(files):
        try:
            logger.info(f"Parsing document : {file} (idx: {idx + 1}/{total})")

            text_splitter_type = None

            if file.endswith(".pdf"):
                logger.info("-> PDF file detected")

                loader = PyPDFLoader(file)
                document = loader.load()

                # Open the PDF file
                with open(file, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    
                    # Extract PDF metadata
                    pdf_info = reader.metadata
                    pdf_title = pdf_info.get('/Title', None)
                    if not pdf_title:
                        pdf_title = "Unknown"
                    document[0].metadata['title'] = pdf_title

                text_splitter_type = "text"
            elif file.endswith(".md"):
                logger.info("-> Markdown file detected")

                loader = UnstructuredMarkdownLoader(file)
                document = loader.load()

                text_splitter_type = "text"
            elif file.endswith(".html"):
                logger.info("-> HTML file detected")

                loader = BSHTMLLoader(file)
                document = loader.load()

                text_splitter_type = "text"
            elif file.endswith(".txt"):
                logger.info("-> Text file detected")

                loader = TextLoader(file, autodetect_encoding=True)
                document = loader.load()

                text_splitter_type = "text"
            elif file.endswith('.odt'):
                logger.info("-> Libre Office file detected")

                loader = UnstructuredODTLoader(file)
                document = loader.load()

                text_splitter_type = "text"
            elif file.endswith('.docx') or file.endswith(".doc"):
                logger.info("-> Microsoft Word file detected")

                loader = Docx2txtLoader(file)
                document = loader.load()

                text_splitter_type = "text"
            elif file.endswith('.pptx') or file.endswith(".ppt"):
                logger.info("-> Microsoft PPT file detected")

                loader = UnstructuredPowerPointLoader(file)
                document = loader.load()

                text_splitter_type = "text"
            else:
                raise Exception(f"Invalid doc type for file '{file}'")

            if text_splitter_type == "text":
                # Split document into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(document)
            else:
                raise Exception(f"Invalid text splitter type '{text_splitter_type}'")
            
            documents.extend(texts)
        except Exception as e:
            logger.error(f"Error parsing {file}: {str(e)}")
            error_count += 1
            continue
        
        # Add documents in batches
        if idx % batch_size == 0 and idx > 0 :
          logger.info(f"Inserting {len(documents)} in database")

          uuids = [str(uuid4()) for _ in range(len(documents))]
          vector_store.add_documents(documents=documents, ids=uuids)
          documents = []

          progression = round(float(idx) / float(total) * 100, 2)
          logger.info(f"Document added : {progression}% ({idx}/{total})")

    # Add any remaining documents
    if documents:
        logger.info(f"Inserting {len(documents)} in database")

        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
    
    logger.info(f"Document added : 100% ({total}/{total})")

    if error_count > 0:
        logger.info(f"{error_count} errors detected during file parsing")

# Interactive query console
def query_documents(vector_store, logger, user_prompt="Query", doc_limit=5):
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        try: 
            user_query = str(query)
            query = build_query(query)
            docs = vector_store.similarity_search(query, k=doc_limit)
            
            display = ""
            display += "\n\n"
            display += "-"*10
            display += "\n"
            display += "Context:"
            display += "\n"
            for idx, doc in enumerate(docs):
                display += "*" * 10
                display += "\n"
                display += f"Quote {idx + 1}:"
                display += "\n"
                display += str(doc.metadata)
                display += "\n"
                display += str(doc.page_content)
                display += "\n"
                display +="*" * 10
                display += "\n\n"
            
            display += f"{user_prompt}: {user_query}"

            logger.info(display)
            pyperclip.copy(display)
            logger.info("\t->Copied to clipboard")
        
        except Exception as e:
            logger.error(f"Error during query execution: {str(e)}")

def initialize_logger(level=logging.INFO, log_file=None):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Prevent logging from propagating to the root logger (avoiding duplicate log messages)
    logger.propagate = False

    # Create a console handler to log to the console (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter and set it for the console handler
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # If a log file is specified, create a file handler to log to the file
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the file handler
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

    return logger


import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Semantic search : query file database using sentence transformers and vector stores")
    
    parser.add_argument(
        '--user_prompt', 
        type=str, 
        default="From documents provided as context, answer to the following query", 
        help="Prompt that is passed to the query console"
    )
    
    parser.add_argument(
        '--doc_limit', 
        type=int, 
        default=5, 
        help="Limit the number of documents to return in response to the query"
    )
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="sentence-transformers/all-mpnet-base-v2", 
        help="Name of the model to use for embedding generation"
    )

    parser.add_argument(
        '--data_dir', 
        type=str, 
        default="./data", 
        help="Path of the file dir"
    )

    parser.add_argument(
        '--log_file', 
        type=str, 
        default="prompt.log", 
        help="Filename for the log file"
    )

    parser.add_argument(
        '--doc_types', 
        type=str, 
        default="pdf", 
        help=f"Extension of document, separated by / (e.g. pdf/md/txt). Supported : {'/'.join(SUPPORTED_DOC_TYPES)}"
    )
    
    args = parser.parse_args()

    logger = initialize_logger(log_file=args.log_file)

    # Use the parsed model_name argument
    embeddings = initialize_embeddings(args.model_name) 
    
    # Initialize vector store with embeddings
    vector_store = initialize_vector_store(embeddings)

    # Load documents from PDFs folder and add them to vector store if needed
    load_and_process_files(args.data_dir, vector_store, logger, doc_types=args.doc_types)
    
    # Start query console using the parsed user_prompt and doc_limit
    query_documents(vector_store, logger, user_prompt=args.user_prompt, doc_limit=args.doc_limit)


if __name__ == '__main__':
    main()
