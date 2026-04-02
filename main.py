from dataclasses import dataclass
from multiprocessing import context

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    explanation: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

@tool('get_weather', description="Get the current weather for a given location.")
def get_weather(city: str) -> str:
    response = requests.get(f"http://wttr.in/{city}?format=j1")
    return response.json()

@tool('locate_user', description="Look up a user's city based on the conext")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XYZ456':
            return 'London'
        case 'HJKL111':
            return 'Paris'
        case _:
            return 'Unknown'
        
model = init_chat_model('mistral-small', temperature=0.3)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt="You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful.",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like?'}
    ]}, 
    config= config,
    context=Context(user_id='XYZ456')
)

print(response['structured_response'])
print(response['structured_response'].explanation)
print(response['structured_response'].temperature_celsius)
print(response['structured_response'].temperature_fahrenheit)
print(response['structured_response'].humidity)



response2 = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like?'}
    ]}, 
    config= config,
    context=Context(user_id='XYZ456')
)

print(response2['structured_response'])
print(response2['structured_response'].explanation)
print(response2['structured_response'].temperature_celsius)
print(response2['structured_response'].temperature_fahrenheit)
