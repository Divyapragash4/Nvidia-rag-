# Standard library imports
import os
import io
import sqlite3
from datetime import datetime

# Google API imports
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle

# Define the scopes your application needs
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Use your existing credentials file
CREDENTIALS_PATH = 'client_secret_358446751792-cj236mannofbcd0e7d920pkj2g73cvgi.apps.googleusercontent.com.json'

def init_database():
    """Initialize SQLite database to store file information"""
    conn = sqlite3.connect('drive_files.db')
    cursor = conn.cursor()
    
    # Create table to store file information
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drive_files (
            file_id TEXT PRIMARY KEY,
            file_name TEXT,
            mime_type TEXT,
            downloaded_path TEXT,
            download_date DATETIME,
            file_size INTEGER
        )
    ''')
    
    conn.commit()
    return conn

def get_drive_service():
    """Initialize Google Drive service with credentials"""
    try:
        creds = None
        # The file token.pickle stores the user's access and refresh tokens
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        print(f"Error initializing Drive service: {e}")
        return None

def list_drive_files(service, page_size=10):
    """List files from Google Drive"""
    try:
        results = service.files().list(
            pageSize=page_size,
            fields="nextPageToken, files(id, name, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print('No files found in Drive.')
            return []
            
        print('\nFiles found in Drive:')
        for file in files:
            print(f"Name: {file['name']}")
            print(f"ID: {file['id']}")
            print(f"Type: {file.get('mimeType', 'Unknown type')}")
            print('-------------------')
            
        return files
    
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

def download_file(service, file_id, file_name, conn):
    """Download a file from Drive and store its info in database"""
    try:
        # Create downloads directory if it doesn't exist
        if not os.path.exists('downloads'):
            os.makedirs('downloads')
            
        # Set up the download path
        download_path = os.path.join('downloads', file_name)
        
        # Get the file from Drive
        request = service.files().get_media(fileId=file_id)
        file_handle = io.BytesIO()
        
        # Download the file
        downloader = MediaIoBaseDownload(file_handle, request)
        done = False
        
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download Progress: {int(status.progress() * 100)}%")
            
        # Save the file locally
        file_handle.seek(0)
        with open(download_path, 'wb') as f:
            f.write(file_handle.read())
            
        # Get file size
        file_size = os.path.getsize(download_path)
        
        # Get file metadata for mime type
        file_metadata = service.files().get(fileId=file_id, fields='mimeType').execute()
        mime_type = file_metadata.get('mimeType', 'unknown')
        
        # Store file information in database
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO drive_files 
            (file_id, file_name, mime_type, downloaded_path, download_date, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (file_id, file_name, mime_type, download_path, datetime.now(), file_size))
        
        conn.commit()
        print(f"\nFile '{file_name}' downloaded successfully to {download_path}")
        
    except Exception as e:
        print(f"Error downloading file: {e}")

def main():
    print("Initializing Google Drive File Manager...")
    
    # Initialize database
    conn = init_database()
    print("Database initialized successfully.")
    
    # Get Drive service
    print("Authenticating with Google Drive...")
    service = get_drive_service()
    if not service:
        print("Failed to initialize Drive service. Please check your credentials.")
        return
    
    print("Successfully connected to Google Drive!")
    
    while True:
        print("\nGoogle Drive File Manager")
        print("========================")
        print("1. List files in Drive")
        print("2. Download a file")
        print("3. View downloaded files")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            print("\nRetrieving files from Drive...")
            files = list_drive_files(service)
            
        elif choice == '2':
            file_id = input("Enter the file ID to download: ")
            file_name = input("Enter the name to save the file as: ")
            print("\nStarting download...")
            download_file(service, file_id, file_name, conn)
            
        elif choice == '3':
            cursor = conn.cursor()
            cursor.execute('SELECT file_name, downloaded_path, download_date, mime_type FROM drive_files')
            files = cursor.fetchall()
            
            if not files:
                print("\nNo files have been downloaded yet.")
            else:
                print("\nDownloaded files:")
                print("================")
                for file in files:
                    print(f"Name: {file[0]}")
                    print(f"Path: {file[1]}")
                    print(f"Downloaded: {file[2]}")
                    print(f"Type: {file[3]}")
                    print('-------------------')
                    
        elif choice == '4':
            print("\nThank you for using Google Drive File Manager!")
            break
            
        else:
            print("\nInvalid choice. Please try again.")
    
    # Close database connection
    conn.close()

if __name__ == '__main__':
    main()
