from dotenv import find_dotenv, load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI
import pandas as pd
from extract_event import (
    nlp_understand_intent as is_calendar_intent,
)
from handle_calendar_request import handle_calendar_request


# Load .env file
load_dotenv(find_dotenv())

# load personal profile into a dataframe
df = pd.read_csv("profile.csv")

# use of a LLM
llm = ChatOpenAI(temperature=0.5, model="gpt-4-1106-preview")

# additional tools for the agent
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# initialize pandas agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type=AgentType.OPENAI_FUNCTIONS,  # type: ignore
    extra_tools=tools,
)

# Define the initial system message with directives for the GPT model
conversation = [
    {
        "role": "system",
        "content": """
            You are a well mannered cheerful assistant named Jarvis for the user. 
            Provide responses in a succinct manner. If you dont know the answer 
            to a question, say you dont have enough information at the moment.
        """,
    }
]


print("hello, i'm your ai assistance, how can i help you?")

# Main loop for the chat
while True:
    # Get user input
    user_input = input("You: ")

    if user_input == "###":
        break  # Exit the chat if the user types ###

    # input needs to be formatted like
    # add an event for my birthday party on 12/24/2024 at 8pm with family
    # and friends at Tony Romma's"
    # TODO: refactor to account for other formats
    if is_calendar_intent(user_input):
        conversation.append(
            {
                "role": "user",
                "content": f"""can you create the event title based on the 
                    intent of {user_input} by removing event details like
                    time, date, location?, do not repeat my question, just 
                    give me the title""",
            }
        )
        event_desc = agent.run(conversation)
        handle_calendar_request(user_input, event_desc=event_desc)
    else:
        # Append the user message to the conversation
        conversation.append({"role": "user", "content": user_input})
        # Generate a response using the GPT-4 model
        completion = agent.run(conversation)
        # Append the AI's response to the conversation
        conversation.append({"role": "assistant", "content": completion})
        # Print the AI's response
        print(f"Assistant: {completion}")
