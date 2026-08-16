# ==============================
# 1. LOAD LIBRARIES
# ==============================

from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage
from tavily import TavilyClient
from langchain.agents import create_agent

# ==============================
# 2. WEATHER TOOL
# ==============================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""

    API_KEY = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )

    response = requests.get(url)

    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# ==============================
# 3. TEST WEATHER TOOL
# ==============================

city = input("Give the city: ")

print(get_weather.invoke({"city": city}))


# ==============================
# 4. TAVILY NEWS TOOL
# ==============================

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get the latest news of this city."""

    query = f"latest news in {city}"

    response = tavily_client.search(
        query=query,
        topic="news",
        time_range="week",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found in {city}."

    news_list = []

    for r in results:

        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n"
            f"  {url}\n"
            f"  {snippet[:100]}..."
        )

    return (
        f"Latest news in {city}:\n\n"
        + "\n\n".join(news_list)
    )


# ==============================
# 5. INITIALIZE LLM
# ==============================

llm = ChatMistralAI(
    model="mistral-small-latest"
)
agent=create_agent(
    llm,
    tools=[get_weather,get_news],
    system_prompt="you are helpful city assistant."
)
print("city agent|type esit to quit")

while True:
    user_input=input("You :")
    if user_input.lower()=="exit":
        break
    result=agent.invoke({
        "messages":[{"role":"user","content":user_input}]
    })
    print("bot :",result["messages"][-1].content)


