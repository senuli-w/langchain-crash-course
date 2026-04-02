import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model="mistral-small",
    temperature=0.1,
)

conversation = [
    SystemMessage("You are a helpful assistant that provides information about programming languages."),
    HumanMessage("Hello, what is Python?"),
    AIMessage("Python is a high-level, interpreted programming language known for its simplicity and readability. It is widely used for web development, data analysis, artificial intelligence, scientific computing, and more. Python has a large standard library and a vibrant community that contributes to its extensive ecosystem of third-party packages."),
    HumanMessage("When was it released?"),
]

response = model.invoke(conversation)

# print(response)
print(response.content)