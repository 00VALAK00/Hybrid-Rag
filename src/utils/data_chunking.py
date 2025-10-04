from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import logging

logger = logging.getLogger(__file__)

data_directory = Path(__file__).resolve().parent.parent / "data"

def load_and_chunk_pdfs(files_path=data_directory, chunk_size=1000, chunk_overlap=200) -> list[dict]:
     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
     for pdf in data_directory.glob("*.pdf"):
        logger.info(f"Document {pdf.name} is being loaded and processed")
        pdfLoader = PyPDFLoader(str(pdf))
        document = pdfLoader.load()
        chunks = text_splitter.split_documents(document)
        logger.info(f"Success")
        
        payloads = []
        for chunk in chunks:

            # Adjusted page number to start from 1 instead of 0
            payload = {"content": chunk.page_content, "source": pdf.name, "page": chunk.metadata.get("page", -1)+1}
            payloads.append(payload)
        return payloads
     
     

    