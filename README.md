# PDF to Markdown Converter

This project provides a FastAPI-based service for converting PDF documents into structured Markdown format using Azure OpenAI and Azure Document Intelligence. It supports parallel processing for efficient conversion and includes features like image compression, retry mechanisms, and webhook integration.

## Features

- **PDF to Markdown Conversion**: Extracts content from PDFs and converts it into Markdown format.
- **Azure OpenAI Integration**: Uses Azure OpenAI for GPT-based content extraction.
- **Azure Document Intelligence**: Leverages Azure Document Intelligence for prebuilt-layout analysis.
- **Image Processing**: Converts PDF pages to images, compresses them, and enhances contrast for better OCR results.
- **Parallel Processing**: Utilizes multithreading for faster processing of large PDFs.
- **Webhook Support**: Sends results to a specified webhook URL upon completion.
- **Error Handling**: Includes retry mechanisms with exponential backoff for rate-limited API calls.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Miracle-Go-123/pdf2markdown.git
   cd pdf2markdown
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Create a `.env` file in the roor directory and configure the following variables.
   ```bash
   OCR_AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
   OCR_AZURE_OPENAI_KEY=<your-azure-openai-key>
   OCR_AZURE_OPENAI_API_VERSION=<api-version>
   OCR_AZURE_DEPLOYMENT_NAME=<deployment-name>
   DI_AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
   DI_AZURE_OPENAI_KEY=<your-azure-openai-key>
   DI_AZURE_OPENAI_API_VERSION=<api-version>
   DI_AZURE_DEPLOYMENT_NAME=<deployment-name>
   AZURE_DOCUMENT_ENDPOINT=<your-document-intelligence-endpoint>
   AZURE_DOCUMENT_KEY=<your-document-intelligence-key>
   NEXT_API_KEY=<your-api-key>
   ```
4. Run the application
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- `GET /`: Health check endpoint.
- `POST /kickoff`: Upload a PDF file for conversion. Returns the converted Markdown content.
- `POST /kickoff_hook`: Upload a PDF file and specify a webhook URL to receive the results asynchronously.
- `GET /status/{job_id}`: Check the status of a conversion job.

## Example Request

```bash
curl -X POST "http://localhost:8000/kickoff" \
  -F "file=@example.pdf"
```

## Configuration

- **Image Compression**: Configurable via `MAX_IMAGE_SIZE_MB` and `TARGET_IMAGE_SIZE_MB` in `config.py`.
- **Threading**: Adjust `MAX_THREADS` for parallel processing.
- **Retry Mechanism**: Customize `RATE_LIMIT_RETRY_MAX_COUNT` and `RATE_LIMIT_RETRY_DELAY` for API rate limits.

## Dependencies

- Python 3.12
- FastAPI
- PyPDF2
- pdf2image
- Azure OpenAI
- Azure Document Intelligence
- Pillow
- psutil
