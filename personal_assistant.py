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
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4-1106-preview",
    response_format={"type": "json_object"},
)

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
            Provide responses in a succinct manner. Provide all your outputs in json format. If you dont know the answer 
            to a question, search the internet, if you're still unsure, say you dont have enough information to answer the question or 
            the ability to complete the task.
            
            - classify each query into two "intent" categories and set the value to either "general" or "calendar"
            - create a "message" category and set the value to any messages or prompts
            - if the intent type is of calendar, create a calendar category using a structured data set for each of the 3 calendar categories:
                add an event: title (required), start-time (required), start-date (required), end-time (optional), end-date (optional), summary (optional), notes (optional), contacts (optional), should-notify-contact (optional), set-reminder (optional), location (optional)
                    - summarize the event in 12 words or less for the summary field if none is provided
                    - default the end-time and end-date to start-time and end-date respectively if none is provided
                    - default the title field to the event if none is given specifically
                    - use mm/dd/yyyy for the date format
                    - use "0:00 PM" for the time format
                    - the contacts field is a list, it is optional and intended for email address(es), do not assume names or person given for the contacts field
                    - both set-reminder and should-notify-contact fields are of data type boolean and have false set as the default
                    - all fields in the add an event category should be present, if no value is defaulted to a field or specified by the user, assign the field to an empty string or empty list
                    - if required fields are missing, prompt the user to provide the information
                    - only if all required fields are filled, prompt the user to confirm all required fields and add to any optional fields if none are provided
                        - for example, "please confirm, for [calendar category] to the calendar, the [event]"
                        - provide a helpful reminder for any optional [key]s left empty
                delete an event: event-id (required)
                update an event: event-id (required), title (optional), start-time (optional), start-date (optional), end-time (optional), end-date (optional), summary (optional), notes (optional), contacts (optional), should-notify-contact (optional), set-reminder (optional) 
            
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
    else:
        # Append the user message to the conversation
        conversation.append({"role": "user", "content": user_input})
        # Generate a response using the GPT-4 model
        completion = agent.run(conversation)
        # Append the AI's response to the conversation
        conversation.append({"role": "assistant", "content": completion})
        # Print the AI's response
        print(f"Assistant: {completion}")
