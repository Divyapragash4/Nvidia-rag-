import streamlit as st
import os
import pdfocr
import embedding
from VectorDB import VectorStore
import drive  # Import the drive module for Google Drive integration
from mistral import client, model  # Import Mistral client and model

# Ensure necessary folders exist
os.makedirs('downloads', exist_ok=True)
os.makedirs('extracted_text', exist_ok=True)
os.makedirs('chunked_texts', exist_ok=True)
os.makedirs('faiss_db', exist_ok=True)

def process_file(file_path):
    # Process PDF OCR
    st.write("Processing PDF for text extraction...")
    text = pdfocr.extract_text_from_pdf(file_path)
    if not text:
        text = pdfocr.extract_text_with_ocr(file_path)
    if text:
        # Save extracted text
        extracted_path = os.path.join('extracted_text', f"{os.path.splitext(os.path.basename(file_path))[0]}_text.txt")
        with open(extracted_path, 'w', encoding='utf-8') as f:
            f.write(text)
        st.success("Text extracted successfully!")

        # Process embedding
        st.write("Processing embeddings...")
        processed = embedding.process_all_texts()
        embedding.save_chunked_texts(processed)
        st.success("Embeddings processed successfully!")

        # Load into vector DB
        st.write("Loading into vector database...")
        vector_store = VectorStore()
        vector_store.load_embeddings()
        st.success("Vector database updated successfully!")
    else:
        st.error("Could not extract text from PDF.")

def process_with_mistral(chunks, query):
    """Process relevant chunks with Mistral LLM"""
    try:
        # Combine chunks into context
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk['text']}" for i, chunk in enumerate(chunks)])
        
        # Create prompt for Mistral
        prompt = f"""Based on the following context and query, provide a comprehensive answer.
        
Query: {query}

Context:
{context}

Please provide a detailed answer that synthesizes information from the relevant chunks."""

        # Get response from Mistral
        response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error processing with Mistral: {str(e)}")
        return None

st.title("PDF Document Processing and Query Interface")

# Initialize Google Drive connection
drive_service = drive.get_drive_service()

# Sidebar for document selection
st.sidebar.title("Available Documents")

# Get local documents
local_docs = [f for f in os.listdir('downloads') if f.endswith('.pdf')]
# Get Google Drive documents
drive_files = drive.list_drive_files(drive_service) if drive_service else []
drive_docs = [f['name'] for f in drive_files if f['name'].lower().endswith('.pdf')]

# Create tabs for different document sources
tab1, tab2 = st.sidebar.tabs(["Local Documents", "Google Drive"])

with tab1:
    st.write("### Local Documents")
    if local_docs:
        selected_local = st.selectbox("Select a local document:", local_docs)
        if selected_local:
            st.write(f"Selected: {selected_local}")
    else:
        st.write("No local documents available")

with tab2:
    st.write("### Google Drive Documents")
    if drive_service:
        if drive_docs:
            selected_drive = st.selectbox("Select a Google Drive document:", drive_docs)
            if selected_drive:
                st.write(f"Selected: {selected_drive}")
        else:
            st.write("No documents in Google Drive")
    else:
        st.write("Google Drive not connected")

# Main content area
st.write("## Upload or Process Documents")

# File uploader
uploaded_file = st.file_uploader("Upload a new PDF file", type=['pdf'])

# Process button for selected documents
if st.button("Process Selected Document"):
    if uploaded_file is not None:
        # Process uploaded file
        file_path = os.path.join('downloads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        process_file(file_path)
    elif selected_local:
        # Process selected local file
        file_path = os.path.join('downloads', selected_local)
        process_file(file_path)
    elif selected_drive and drive_service:
        # Process selected Google Drive file
        file_id = next((f['id'] for f in drive_files if f['name'] == selected_drive), None)
        if file_id:
            conn = drive.init_database()
            file_path = drive.download_file(drive_service, file_id, selected_drive, conn)
            if file_path:
                process_file(file_path)
            else:
                st.error("Failed to download file from Google Drive")
        else:
            st.error("Could not find file ID for selected document")
    else:
        st.warning("Please select or upload a document to process")

# Query interface
st.write("## Query Interface")
query = st.text_input("Enter your query:")
if query:
    vector_store = VectorStore()
    vector_store.load_embeddings()
    results = vector_store.query(query, n_results=5)
    if results:
        st.write("### Query Results:")
        
        # Display raw chunks
        st.write("#### Relevant Chunks:")
        for result in results:
            st.write(f"**Text:** {result['text'][:200]}...")
            st.write(f"**Distance:** {result['distance']}")
            st.write(f"**Source:** {result['metadata']['source']}")
            st.write("---")
        
        # Process with Mistral
        if st.button("Process with Mistral"):
            with st.spinner("Processing with Mistral..."):
                mistral_response = process_with_mistral(results, query)
                if mistral_response:
                    st.write("#### Mistral Analysis:")
                    st.write(mistral_response)
    else:
        st.write("No results found for your query.")
