import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool('get_weather', description='Return weather information for a given location')
def get_weather(city: str) -> str:
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    data = response.json()
    
    # Extract relevant weather info
    current = data.get('current_condition', [{}])[0]
    location = data.get('nearest_area', [{}])[0]
    
    weather_info = f"""Weather in {location.get('areaName', [{}])[0].get('value', 'Unknown')}, {location.get('country', [{}])[0].get('value', 'Unknown')}:
- Temperature: {current.get('temp_C')}°C ({current.get('temp_F')}°F)
- Condition: {current.get('weatherDesc', [{}])[0].get('value', 'Unknown')}
- Humidity: {current.get('humidity')}%
- Wind: {current.get('windspeedKmph')} km/h
- Feels Like: {current.get('FeelsLikeC')}°C"""
    
    return weather_info

agent = create_agent(
    model='mistral-small', 
    tools=[get_weather], 
    system_prompt="You are a helpful weather assistant. who always cracks jokes and is humorous while remaining helpful."
)

output = agent.invoke({
    'messages':[
        {'role': 'user', 'content': 'What is the weather like in Vienna?'}
    ]
})

print(output)
print(output['messages'][-1].content)