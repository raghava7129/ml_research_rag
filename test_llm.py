import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("GOOGLE_API_KEY not found in .env")
    exit(1)

print(f"API key loaded")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    response = llm.invoke("Say 'API key works!' and nothing else.")
    print(f"Gemini response: {response.content}")

except Exception as e:
    print(f"API call failed: {e}")