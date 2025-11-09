from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from pypdf import PdfReader
from dotenv import load_dotenv
from chat1_handler import handle_chat1_request

# Load environment variables - try production first, then fallback to .env
env_file = ".env.production" if os.path.exists(".env.production") else ".env"
load_dotenv(env_file, override=True)

# Get max file size from environment (default 100MB)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Configure FastAPI
# Note: For large file uploads, the request body size limit is controlled by:
# 1. Uvicorn server configuration (see server.sh)
# 2. Reverse proxy (nginx) if used (see nginx_config_example.conf)
# 3. Client-side limits
app = FastAPI()

# Allow CORS (so Next.js can talk to FastAPI)
# Get allowed origins from environment variable, default to "*" for development
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
if allowed_origins == ["*"]:
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in allowed_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
gemini_api_key=os.getenv("GOOGLE_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
client = OpenAI(base_url=GEMINI_BASE_URL, api_key=gemini_api_key)

###########################################################################################
# my personal assistant                                                                   #
###########################################################################################
@app.post("/chat")
async def chat(message: str = Form(None), file: UploadFile = File(None)):
    name = os.getenv("NAME")
    #reader = PdfReader("me/linkedin.pdf")
    reader = PdfReader("me/Profile.pdf")
    linkedin = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            linkedin += text
    with open("me/summary.txt", "r", encoding="utf-8") as f:
        summary = f.read()

    #define system prompt
    system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
    particularly questions related to {name}'s career, background, skills and experience. \
    Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
    You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
    Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
    If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
    If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

    system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
    system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

    response_text = ""

    # Optionally read file content
    if file:
        file_content = await file.read()
        # Process the file if needed (PDF parsing, etc.)
        response_text += f"Received file: {file.filename}. "

    # Send to AI (optional, replace with your logic)
    if message:
        ai_response = client.chat.completions.create(
            model="gemini-2.5-flash-preview-05-20",  # or another model
            messages=[{"role": "system", "content": system_prompt}] + 
            [{"role": "user", "content": message}]
            
        )
        response_text += ai_response.choices[0].message.content
    else:
        response_text += "I received your file!"

    return {"reply": response_text}
###########################################################################################
# CHAT1 - Vector DB with PDF/Text input and query support                                 #
###########################################################################################
@app.post("/chat1")
async def chat1(
    select: str = Form(...), 
    message: str = Form(...), 
    file: UploadFile = File(None), 
    model: str = Form(None),
    operation: str = Form(None)
):
    """
    Endpoint for processing documents and querying vector database.
    
    Parameters:
    - select: "input", "query", or "data" - determines the operation type
    - message: The actual content/text/chunk_id
    - file: Optional PDF file (for input operation)
    - model: Optional model selection ("ollama", "gemma3:12b", or None for Gemini default)
    - operation: Optional operation for data select ("get" or "delete")
    
    Operations:
    - select="input": Process PDF file or text, extract/summarize via LLM, chunk and store in ChromaDB
    - select="query": Retrieve relevant context from vector DB and get answer from LLM
    - select="data": Get or delete chunks from ChromaDB
      - operation="get" with message="all" or message="chunk_id": Get all chunks or specific chunk
      - operation="delete" with message="all" or message="chunk_id": Delete all chunks or specific chunk
    
    Model selection (default: gemini-2.5-flash-preview-05-20):
    - Pass 'model' parameter: "ollama" or "gemma3:12b" to use Ollama, or omit/None/empty for Gemini default
    """
    return await handle_chat1_request(select, message, file, gemini_api_key, model, operation)


###########################################################################################
# CHAT2 - Industry-Standard Modular RAG Pipeline                                          #
###########################################################################################
@app.post("/chat2")
async def chat2(
    select: str = Form(...), 
    message: str = Form(...), 
    file: UploadFile = File(None), 
    model: str = Form(None),
    operation: str = Form(None)
):
    """
    Endpoint for modular RAG pipeline operations.
    
    Parameters:
    - select: "input", "query", or "data" - determines the operation type
    - message: The actual content/text/chunk_id
    - file: Optional PDF file (for input operation)
    - model: Optional model selection ("ollama", "gemma3:12b", or None for Gemini default)
    - operation: Optional operation for data select ("get" or "delete")
    
    Operations:
    - select="input": Process PDF/text through ingestion -> embedding -> vector store pipeline
    - select="query": Retrieve relevant context and generate answer via LLM
    - select="data": Get, delete, or similarity search chunks from vector database
      - operation="get": Get all chunks or specific chunk
      - operation="delete": Delete all chunks or specific chunk
      - operation="similarity": Find top 20 most similar chunks to query message
    """
    from chat2_handler import handle_chat2_request
    return await handle_chat2_request(select, message, file, gemini_api_key, model, operation)
