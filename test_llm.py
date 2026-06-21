# import os
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.getenv("GOOGLE_API_KEY")

# if not api_key:
#     print("GOOGLE_API_KEY not found in .env")
#     exit(1)

# print(f"API key loaded")

# try:
#     from langchain_google_genai import ChatGoogleGenerativeAI

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-3.5-flash",
#         google_api_key=api_key,
#         temperature=0,
#     )

#     response = llm.invoke("Say 'API key works!' and nothing else.")
#     print(f"Gemini response: {response.content}")

# except Exception as e:
#     print(f"API call failed: {e}")



import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain.community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    ChatPromptTemplate,
)

examples = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 5+3?", "answer": "8"},
]

example_prompt = PromptTemplate(
    template="Question: {question}\nAnswer: {answer}",
    input_variables=["question", "answer"],
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Answer the math question following these examples:",
    suffix="",
    input_variables=[],
)

few_shot_text = few_shot_prompt.format()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", few_shot_text),
    ("human", "{question}"),
])

messages = chat_prompt.format_messages(question="What is 9+7?")
for m in messages:
    print(m.type, ":", m.content)