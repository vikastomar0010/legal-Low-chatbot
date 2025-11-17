import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import create_chunks

create_chunks(
    txt_dir="backend/data/indian-law",
    pdf_dir="backend/data/indian-law",
    output_path="backend/data/legal_chunks.npy"
)

