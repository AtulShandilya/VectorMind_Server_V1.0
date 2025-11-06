# LLM 3.0 - FastAPI Vector Database & LLM Integration

A FastAPI-based application that provides intelligent document processing and querying capabilities using vector databases (ChromaDB) and Large Language Models (Gemini 2.5 Flash or Ollama). The system can process PDF documents, extract and summarize content, chunk it with metadata, store it in a vector database, and provide intelligent querying capabilities.

## Features

- 📄 **PDF & Text Processing**: Extract text from PDF files or process plain text input
- 🤖 **Multi-LLM Support**: Choose between Gemini 2.5 Flash (default) or Ollama (gemma3:12b)
- 📊 **Vector Database Integration**: Store and retrieve document chunks using ChromaDB
- 🔍 **Semantic Search**: Query documents using vector similarity search
- 📝 **Intelligent Chunking**: Automatically chunk documents with context-aware overlaps (2000 chars with 300 char overlap)
- 🗂️ **Metadata Management**: Store and retrieve chunks with rich metadata
- 💬 **Personal Assistant**: Built-in chat endpoint for personal assistant functionality
- 🔧 **Data Management**: Get, delete, and manage chunks in the vector database

## Requirements

- Python 3.11+
- Ubuntu Linux (tested on 2GB RAM servers)
- Google API Key (for Gemini) or Ollama running locally (for Ollama model)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd LLM_3.0
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install CPU-Only PyTorch (Recommended for 2GB RAM)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

Create a `.env` file or `.env.production` file in the project root:

```env
# Gemini API Configuration
GOOGLE_API_KEY=your_gemini_api_key_here

# Personal Assistant Configuration (for /chat endpoint)
NAME=Your Name

# CORS Configuration (optional)
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
# Or use "*" for development (allows all origins)

# ChromaDB Configuration (optional)
CHROMA_DIR=./chroma_db
CHROMA_COLLECTION_NAME=chat1_collection

# Embedding Model Configuration (optional, defaults to all-MiniLM-L6-v2)
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

### 6. Set Up Ollama (Optional, for using Ollama model)

If you want to use Ollama instead of Gemini:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the gemma3:12b model
ollama pull gemma3:12b

# Start Ollama server (usually runs on http://localhost:11434)
ollama serve
```

## Running the Application

<<<<<<< HEAD
=======
#Important #
please add a folder ./me and add a profile.pdf or remove the section /chat and keep only /chat1 in server.py

>>>>>>> 784733be6fd32e5bc1ba76f729a190b6d231e23e
### Development Mode

```bash
source venv/bin/activate
<<<<<<< HEAD
uvicorn chatServer:app --host 0.0.0.0 --port 8000 --reload
=======
uvicorn Server:app --host 0.0.0.0 --port 8000 --reload
>>>>>>> 784733be6fd32e5bc1ba76f729a190b6d231e23e
```

### Production Mode (using PM2)

```bash
source venv/bin/activate
<<<<<<< HEAD
pm2 start "uvicorn chatServer:app --host 0.0.0.0 --port 8000" --name fastapi-app
=======
pm2 start "uvicorn Server:app --host 0.0.0.0 --port 8000" --name fastapi-app
>>>>>>> 784733be6fd32e5bc1ba76f729a190b6d231e23e
```

Or use the provided `server.sh` script:

```bash
chmod +x server.sh
./server.sh
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. `/chat` - Personal Assistant

Chat endpoint that acts as a personal assistant based on your profile.

**Endpoint:** `POST /chat`

**Parameters:**
- `message` (string, optional): User message
- `file` (file, optional): Optional file upload

**Example:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -F "message=Tell me about your experience"
```

---

### 2. `/chat1` - Vector Database Operations

Main endpoint for document processing, querying, and data management.

**Endpoint:** `POST /chat1`

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `select` | string | Yes | Operation type: `"input"`, `"query"`, or `"data"` |
| `message` | string | Yes | Content/text/chunk_id based on operation |
| `file` | file | No | PDF file (for `select="input"`) |
| `model` | string | No | Model selection: `"ollama"`, `"gemma3:12b"`, or `None` (Gemini default) |
| `operation` | string | No | For `select="data"`: `"get"` or `"delete"` |

#### Operations

##### A. Input Operation (`select="input"`)

Process PDF files or text, extract/summarize via LLM, chunk into 2000-character pieces with 300-character overlap, and store in ChromaDB.

**Example with PDF:**
```bash
curl -X POST "http://localhost:8000/chat1" \
  -F "select=input" \
  -F "message=Process this document" \
  -F "file=@document.pdf" \
  -F "model=ollama"
```

**Example with Text:**
```bash
curl -X POST "http://localhost:8000/chat1" \
  -F "select=input" \
  -F "message=Your text content here"
```

**Response:**
```json
{
  "status": "success",
  "message": "Document processed and stored successfully",
  "ingested_chunks": 5,
  "document_summary": "Comprehensive summary...",
  "chunks_count": 5
}
```

##### B. Query Operation (`select="query"`)

Retrieve relevant context from ChromaDB and get an answer from the LLM.

**Example:**
```bash
curl -X POST "http://localhost:8000/chat1" \
  -F "select=query" \
  -F "message=What are the key findings in the document?" \
  -F "model=ollama"
```

**Response:**
```json
{
  "status": "success",
  "query": "What are the key findings in the document?",
  "answer": "Based on the context...",
  "retrieved_chunks": 5,
  "sources": [
    {
      "chunk_index": 1,
      "metadata": {...},
      "relevance_score": 0.95
    }
  ]
}
```

##### C. Data Operations (`select="data"`)

Manage chunks in ChromaDB.

**Get All Chunks:**
```bash
curl -X POST "http://localhost:8000/chat1" \
  -F "select=data" \
  -F "operation=get" \
  -F "message=all"
```

**Get Specific Chunk:**
```bash
curl -X POST "http://localhost:8000/chat1" \
  -F "select=data" \
  -F "operation=get" \
  -F "message=chunk_id_123"
```

**Delete All Chunks:**
```bash
curl -X POST "http://localhost:8000/chat1" \
  -F "select=data" \
  -F "operation=delete" \
  -F "message=all"
```

**Delete Specific Chunk:**
```bash
curl -X POST "http://localhost:8000/chat1" \
  -F "select=data" \
  -F "operation=delete" \
  -F "message=chunk_id_123"
```

## Model Selection

### Gemini (Default)

- **Model:** `gemini-2.5-flash-preview-05-20`
- **Usage:** Omit `model` parameter or set to `None`
- **Requires:** `GOOGLE_API_KEY` environment variable

### Ollama

- **Model:** `gemma3:12b`
- **Usage:** Set `model="ollama"` or `model="gemma3:12b"`
- **Requires:** Ollama server running on `http://localhost:11434`

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Gemini API key | Required |
| `NAME` | Name for personal assistant | Required for `/chat` |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | `*` |
| `CHROMA_DIR` | ChromaDB storage directory | `./chroma_db` |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection name | `chat1_collection` |
| `EMBEDDING_MODEL_NAME` | Sentence transformer model | `all-MiniLM-L6-v2` |

### Chunk Configuration

- **Chunk Size:** 2000 characters
- **Overlap Size:** 300 characters
- **Embedding Model:** `all-MiniLM-L6-v2` (lightweight, suitable for 2GB RAM)

## Deployment on Ubuntu (2GB RAM)

### 1. Install Python 3.11

```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv python3.11-dev
```

### 2. Set Up Application

```bash
# Clone repository
git clone <your-repo-url>
cd LLM_3.0

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install CPU-only PyTorch (important for low RAM)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Create .env.production file
nano .env.production

# Add your configuration
GOOGLE_API_KEY=your_api_key
NAME=Your Name
ALLOWED_ORIGINS=https://yourdomain.com
```

### 4. Run with PM2

```bash
# Install PM2
npm install -g pm2

# Start application
source venv/bin/activate
<<<<<<< HEAD
pm2 start "uvicorn chatServer:app --host 0.0.0.0 --port 8000" --name fastapi-app
=======
pm2 start "uvicorn Server:app --host 0.0.0.0 --port 8000" --name fastapi-app
>>>>>>> 784733be6fd32e5bc1ba76f729a190b6d231e23e

# Save PM2 configuration
pm2 save

# Setup PM2 to start on boot
pm2 startup
```

### 5. Monitor Application

```bash
# View logs
pm2 logs fastapi-app

# Check status
pm2 status

# Restart application
pm2 restart fastapi-app
```

## Project Structure

```
LLM_3.0/
<<<<<<< HEAD
├── chatServer.py          # Main FastAPI application
=======
├── Server.py              # Main FastAPI application
>>>>>>> 784733be6fd32e5bc1ba76f729a190b6d231e23e
├── chat1_handler.py       # Handler for /chat1 endpoint operations
├── requirements.txt       # Python dependencies
├── server.sh              # PM2 startup script
├── .env                   # Environment variables (development)
├── .env.production        # Environment variables (production)
├── chroma_db/             # ChromaDB storage directory
├── me/                    # Personal assistant resources
│   ├── Profile.pdf        # Profile PDF
│   └── summary.txt        # Profile summary
└── venv/                  # Virtual environment
```

## Memory Optimization

For servers with limited RAM (2GB), the following optimizations are implemented:

1. **CPU-only PyTorch:** Reduces memory footprint
2. **Lightweight Embedding Model:** Uses `all-MiniLM-L6-v2` (22MB)
3. **Efficient ChromaDB:** Uses DuckDB+Parquet for storage
4. **Text Extraction:** Processes PDFs in chunks to avoid loading entire files

## Troubleshooting

### Common Issues

1. **Out of Memory:**
   - Ensure CPU-only PyTorch is installed
   - Use the lightweight embedding model (default)
   - Reduce chunk size if needed

2. **ChromaDB Errors:**
   - Check write permissions for `chroma_db` directory
   - Ensure sufficient disk space

3. **Ollama Connection Errors:**
   - Verify Ollama is running: `curl http://localhost:11434/api/tags`
   - Check firewall settings

4. **Gemini API Errors:**
   - Verify `GOOGLE_API_KEY` is set correctly
   - Check API quota limits

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]

## Author

[Your Name]

## Acknowledgments

- FastAPI for the web framework
- ChromaDB for vector database
- Google Gemini for LLM capabilities
- Ollama for local LLM support
- Sentence Transformers for embeddings

<<<<<<< HEAD

=======
>>>>>>> 784733be6fd32e5bc1ba76f729a190b6d231e23e
