import os
import phi
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from phi.playground import Playground, serve_playground_app

phi.api = os.getenv("PHI_API_KEY")

web_search_agent = Agent(
    name="Web Search Agent",
    role="You are an agent that helps users find the latest information.",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[GoogleSearch()],
    instructions=["Always include sources in the search results"],
    show_tool_calls=True,
    markdown=True
)

financial_agent = Agent(
    name="Financial AI Agent",
    role="This agent provides financial information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
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

app = Playground(agents=[financial_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
