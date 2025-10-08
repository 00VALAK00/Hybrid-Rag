from langchain.chat_models import init_chat_model
import os 
from dotenv import load_dotenv
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(str(root_dir / ".env") )


    
API_KEY = os.getenv("GEMINI_API","")
llm = init_chat_model(model_provider="google_genai",
                        model= "gemini-2.0-flash",
                        temperature=0.7, 
                        max_retries=3,
                        api_key=API_KEY
                        )
