from retriever import kb_retriever
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from prompts import ORGANIC_PROMPT
import ollama
import os
import google.generativeai as genai


load_dotenv() # Load .env file

llm_model_name = os.getenv("LLM_MODEL_NAME")
gemini_api_ley = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize Gemini client
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
client = genai.GenerativeModel('gemini-1.5-flash')

# API request model
class QueryData(BaseModel):
    query: str

def fetch_context(query: str) -> dict:
    docs = kb_retriever.retrieve(query)
    context = []
    for doc in docs:
        context.append(
            {
                "text" : doc.node.text,
                "score": doc.score
            }
        )
    return context

def fetch_bot_response(query, context):
    print(context)
    context = "".join(f"{i['text']}\n" for i in context)
    prompt = ORGANIC_PROMPT.format(context, query)

    response = client.generate_content(prompt)
    return response.text

@app.post("/get-response/")
async def get_response(query_info: QueryData):
    context = fetch_context(query_info.query)
    answer = fetch_bot_response(query_info.query, context)
    response = {
        "answer": answer,
        "context": context,
    }
    return response