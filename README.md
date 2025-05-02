# Document Processing and Query System

A system for processing PDF documents, extracting text, creating embeddings, and querying content using vector search and LLM analysis.

## Features

- PDF text extraction and OCR
- Google Drive integration
- Text chunking and embedding
- Vector database search
- Mistral LLM integration for analysis
- Streamlit web interface

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and configuration

## Configuration

Create a `.env` file with the following variables:

```env
# Google Drive Configuration
GOOGLE_DRIVE_CLIENT_SECRET_FILE=your_client_secret.json
GOOGLE_DRIVE_SCOPES=https://www.googleapis.com/auth/drive.readonly

# Mistral Configuration
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-large-latest

# Directory Configuration
DOWNLOADS_DIR=downloads
EXTRACTED_TEXT_DIR=extracted_text
CHUNKED_TEXTS_DIR=chunked_texts
FAISS_DB_DIR=faiss_db
```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload or select PDF documents
3. Process documents through the pipeline
4. Query the processed documents
5. Use Mistral for advanced analysis

## Security Notes

- Never commit sensitive files to the repository
- Keep API keys and credentials in `.env` file
- The following files are ignored by Git:
  - `client_secret_*.json`
  - `token.pickle`
  - `.env`
  - Generated data directories

## License

[Your License Here] 