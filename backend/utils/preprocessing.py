from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import numpy as np
from tqdm import tqdm
import os, glob

def create_chunks(txt_dir=None, pdf_dir=None, output_path="backend/data/legal_chunks.npy", max_txt=None, max_pdf=1):
    """
    Loads text and PDF files, splits them into chunks, and saves as .npy file.
    Shows progress bars for file loading and chunk creation.
    """
    print("🧭 Current working directory:", os.getcwd())
    print("📁 TXT path:", txt_dir)
    print("📁 PDF path:", pdf_dir)

    txt_files = glob.glob(os.path.join(txt_dir, "*.txt")) if txt_dir else []
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf")) if pdf_dir else []
    print(f"🔍 Found {len(txt_files)} TXT files and {len(pdf_files)} PDF files.")
    
    all_documents = []

    print("🔹 Starting document loading...")

    # -------------------------------------------------------------------------
    # Load TXT documents
    # -------------------------------------------------------------------------
    if txt_dir and os.path.exists(txt_dir):
        txt_loader = DirectoryLoader(txt_dir, glob="*.txt")
        txt_documents = txt_loader.load()[:max_txt]
        print(f"✅ Loaded {len(txt_documents)} text documents from {txt_dir}")
        all_documents.extend(txt_documents)
    else:
        print("⚠️ Skipping TXT: No valid txt_dir provided or folder not found.")

    # -------------------------------------------------------------------------
    # Load PDF documents (limited properly)
    # -------------------------------------------------------------------------
    if pdf_dir and os.path.exists(pdf_dir):
        limited_pdfs = pdf_files[:max_pdf]  # load only a few for testing
        pdf_documents = []
        for pdf_path in tqdm(limited_pdfs, desc="📚 Loading PDFs", unit="file"):
            try:
                loader = PyPDFLoader(pdf_path)
                pdf_documents.extend(loader.load())
            except Exception as e:
                print(f"⚠️ Skipping {pdf_path}: {e}")
        print(f"✅ Loaded {len(pdf_documents)} PDF pages from {len(limited_pdfs)} files")
        all_documents.extend(pdf_documents)
    else:
        print("⚠️ Skipping PDF: No valid pdf_dir provided or folder not found.")

    if not all_documents:
        print("❌ No documents found! Exiting.")
        return

    # -------------------------------------------------------------------------
    # Split documents into chunks with progress bar
    # -------------------------------------------------------------------------
    print("🔹 Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

    chunks = []
    for doc in tqdm(all_documents, desc="🧩 Processing documents", unit="doc"):
        chunks.extend(text_splitter.split_documents([doc]))

    print(f"✅ Finished splitting. Total chunks created: {len(chunks)}")

    # -------------------------------------------------------------------------
    # Save chunks
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array([chunk.page_content for chunk in chunks], dtype=object))
    print(f"💾 Chunks saved to: {output_path}")
