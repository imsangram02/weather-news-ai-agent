# ==============================
# 1. LOAD LIBRARIES
# ==============================

from dotenv import load_dotenv
load_dotenv()

import os
import requests
import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage
from tavily import TavilyClient
from langchain.agents import create_agent

# ==============================
# 2. PAGE CONFIG + STYLING
# ==============================

st.set_page_config(
    page_title="City Assistant",
    page_icon="🌆",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    /* Main title */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
        letter-spacing: 0.5px;
    }
    .sub-title {
        text-align: center;
        color: #b8c6db;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.8rem;
    }

    /* Weather card */
    .weather-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.1rem 1.4rem;
        color: #f5f7fa;
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
        font-size: 1.05rem;
    }

    /* Chat bubbles */
    .chat-bubble-user {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: white;
        padding: 0.7rem 1rem;
        border-radius: 16px 16px 2px 16px;
        margin: 0.4rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .chat-bubble-bot {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255,255,255,0.12);
        color: #f0f2f5;
        padding: 0.7rem 1rem;
        border-radius: 16px 16px 16px 2px;
        margin: 0.4rem 0;
        max-width: 80%;
        margin-right: auto;
        backdrop-filter: blur(4px);
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.9);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    .stTextInput input, .stChatInput textarea {
        border-radius: 12px !important;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🌆 City Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Weather updates, local news &amp; a friendly city chatbot</div>',
    unsafe_allow_html=True,
)

# ==============================
# 3. WEATHER TOOL
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
# 5. INITIALIZE LLM + AGENT (cached so it's built once per session)
# ==============================

@st.cache_resource
def get_agent():
    llm = ChatMistralAI(
        model="mistral-small-latest"
    )
    return create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt="you are helpful city assistant."
    )


agent = get_agent()

# ==============================
# 6. SIDEBAR — WEATHER LOOKUP
# ==============================

with st.sidebar:
    st.markdown("### ⛅ Quick Weather Check")
    sidebar_city = st.text_input("Give the city:", key="weather_city")

    if st.button("Check Weather", use_container_width=True):
        if sidebar_city.strip():
            weather_result = get_weather.invoke({"city": sidebar_city})
            st.markdown(
                f'<div class="weather-card">{weather_result}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Please enter a city name.")

    st.markdown("---")
    st.markdown("### 🤖 About")
    st.write("Chat with the city assistant below. Type **exit** to end the conversation.")

# ==============================
# 7. MAIN — CHAT INTERFACE
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_ended" not in st.session_state:
    st.session_state.chat_ended = False

st.markdown("### 💬 City Agent")
st.caption("type exit to quit")

# render chat history
for role, content in st.session_state.messages:
    if role == "user":
        st.markdown(f'<div class="chat-bubble-user">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">{content}</div>', unsafe_allow_html=True)

if st.session_state.chat_ended:
    st.info("Conversation ended. Refresh the page to start a new one.")
else:
    user_input = st.chat_input("You:")

    if user_input:
        st.session_state.messages.append(("user", user_input))

        if user_input.lower() == "exit":
            st.session_state.chat_ended = True
            st.rerun()
        else:
            with st.spinner("Thinking..."):
                result = agent.invoke({
                    "messages": [{"role": "user", "content": user_input}]
                })
                bot_reply = result["messages"][-1].content

            st.session_state.messages.append(("bot", bot_reply))
            st.rerun()