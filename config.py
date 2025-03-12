from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
OCR_AZURE_OPENAI_ENDPOINT = os.getenv('OCR_AZURE_OPENAI_ENDPOINT')
OCR_AZURE_OPENAI_KEY = os.getenv('OCR_AZURE_OPENAI_KEY')
OCR_AZURE_OPENAI_API_VERSION = os.getenv('OCR_AZURE_OPENAI_API_VERSION')
OCR_AZURE_DEPLOYMENT_NAME = os.getenv('OCR_AZURE_DEPLOYMENT_NAME')

DI_AZURE_OPENAI_ENDPOINT = os.getenv('DI_AZURE_OPENAI_ENDPOINT')
DI_AZURE_OPENAI_KEY = os.getenv('DI_AZURE_OPENAI_KEY')
DI_AZURE_OPENAI_API_VERSION = os.getenv('DI_AZURE_OPENAI_API_VERSION')
DI_AZURE_DEPLOYMENT_NAME = os.getenv('DI_AZURE_DEPLOYMENT_NAME')

AZURE_DOCUMENT_ENDPOINT = os.getenv('AZURE_DOCUMENT_ENDPOINT')
AZURE_DOCUMENT_KEY = os.getenv('AZURE_DOCUMENT_KEY')

# Next API Key
NEXT_API_KEY = os.getenv('NEXT_API_KEY')

# Temporary directory for storing PNG files
TEMP_DIR = "temp"

# Image processing settings
MAX_IMAGE_SIZE_MB = 5
TARGET_IMAGE_SIZE_MB = 4.5  # Slightly below max for safety margin

# Threading settings
MAX_THREADS = 50  # Maximum number of concurrent API calls

# Save to markdown file
SAVE_TO_MARKDOWN = os.getenv('SAVE_TO_MARKDOWN', 'False').lower() in ('true', '1')

# Format raw markdown from Document Intelligence
FORMAT_RAW_MARKDOWN_FROM_DI = os.getenv('FORMAT_RAW_MARKDOWN_FROM_DI', 'False').lower() in ('true', '1')

# Retry settings for handling rate limit errors
RATE_LIMIT_RETRY_MAX_COUNT = 5
RATE_LIMIT_RETRY_DELAY = 2

# Chunk size for Document Intelligence
CHUNK_SIZE = 6  # Maximum size in MB per chunk

