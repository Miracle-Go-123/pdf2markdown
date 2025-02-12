from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_DEPLOYMENT_NAME = "gpt-4o-08-06"
AZURE_OPENAI_API_VERSION = "2024-05-01-preview"

# Next API Key
NEXT_API_KEY = os.getenv('NEXT_API_KEY')

# Temporary directory for storing PNG files
TEMP_DIR = "temp"

# Output directory for markdown files
OUTPUT_DIR = "output"

# Image processing settings
MAX_IMAGE_SIZE_MB = 5
TARGET_IMAGE_SIZE_MB = 4.5  # Slightly below max for safety margin

# Threading settings
MAX_THREADS = 50  # Maximum number of concurrent API calls
