import os
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from dotenv import load_dotenv
load_dotenv()

web_search_agent = Agent(
    name="Web Search Agent",
    role="You are an agent that helps users find the latest information.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[GoogleSearch()],
    instructions=["Always include sources in the search results"],
    show_tool_calls=True,
    markdown=True
)

financial_agent = Agent(
    name="Financial AI Agent",
    role="This agent provides financial information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True,
    )],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent = Agent(
    team=[web_search_agent, financial_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "Use the web search agent to find information",
        "Use the financial AI agent to provide financial data"
    ],
    markdown=True,
    show_tool_calls=True
)

multi_ai_agent.print_response(
    "Summarize analyst information and share the latest news for Meta", stream=True)
