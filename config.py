import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Drive Configuration
GOOGLE_DRIVE_CONFIG = {
    'client_secret_file': os.getenv('GOOGLE_DRIVE_CLIENT_SECRET_FILE', 'client_secret_*.json'),
    'scopes': [os.getenv('GOOGLE_DRIVE_SCOPES', 'https://www.googleapis.com/auth/drive.readonly')]
}

# Mistral Configuration
MISTRAL_CONFIG = {
    'api_key': os.getenv('MISTRAL_API_KEY'),
    'model': os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
}

# Directory Configuration
DIRECTORIES = {
    'downloads': os.getenv('DOWNLOADS_DIR', 'downloads'),
    'extracted_text': os.getenv('EXTRACTED_TEXT_DIR', 'extracted_text'),
    'chunked_texts': os.getenv('CHUNKED_TEXTS_DIR', 'chunked_texts'),
    'faiss_db': os.getenv('FAISS_DB_DIR', 'faiss_db')
}

# Create directories if they don't exist
for directory in DIRECTORIES.values():
    os.makedirs(directory, exist_ok=True) 