from phi.agent import Agent
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.model.openai import OpenAIChat
from dotenv import load_dotenv
load_dotenv()

db_url = "postgresql+psycopg://aiproject1:aiproject1@localhost:5532/aiproject1"
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        search_type=SearchType.hybrid
    ),
)
# Load the knowledge base: Comment out after first run
knowledge_base.load(recreate=True, upsert=True)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    knowledge=knowledge_base,
    read_chat_history=True,  # Add a tool to read chat history.
    search_knowledge=True,  # Added only if knowledge is provided. the agent would not be able to answer the user questions without this parameter
    show_tool_calls=True,
    markdown=True
)

agent.print_response(
    "How do I make chicken and galangal in coconut milk soup", stream=True)
agent.print_response(
    "What is Yum Woon Sen Neua?", stream=True)
agent.print_response(
    "What kind of information do you provide", stream=True)
agent.print_response("What was my last question?", stream=True)
