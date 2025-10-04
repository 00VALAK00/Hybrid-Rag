from langchain.chat_models import init_chat_model


llm = init_chat_model(model_provider = "ollama",
                    model= "gemma2:9b",
                    temperature=0.7, 
                    max_retries=3)