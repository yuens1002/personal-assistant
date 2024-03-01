from dotenv import find_dotenv, load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI
import pandas as pd

from create_event import create_event
import json


# Load .env file
load_dotenv(find_dotenv())

# load personal profile into a dataframe
df = pd.read_csv("profile.csv")

# use of a LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4-0125-preview",
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
            Provide responses in json format and in a succinct manner. you're provided some basic information about the user. If you dont know the answer 
            to a question, search the internet, if you're still unsure, say you dont have enough information to answer the question or 
            the ability to complete the task.
            
            - classify each query into two "intent" categories and set the value to type: enum ["general", "calendar"]
            - create a "message" category and set the value to any messages or prompts
            - if the intent type is of calendar, create a category using a structured data set for each of the 3 following categories:
                ["add_an_event"]: 
                title (required), start_time (required), start_date (required), end_time (optional), end_date (optional), summary (optional), attendees (optional), reminders (optional), location (optional), confirmed (default: false)
                    - summarize the event in 12 words or less for the summary field if none is provided
                    - default the end-time and end-date to start-time and end-date respectively if none is provided
                    - default the title field to the event if none is given specifically
                    - use mm/dd/yyyy for the date format
                    - use "0:00 [AM/PM]" for the time format
                    - the [reminders] field is optional with the definition below
                        "reminders": {
                            "type": "array",
                            "items": {
                            "type": "object",
                            "properties": {
                                "method": {
                                "type": "string",
                                "enum": [
                                    "email",
                                    "popup"
                                ]
                                },
                                "minutes": {
                                "type": "number"
                                }
                            },
                            "required": [
                                "method",
                                "minutes"
                            ]
                            }
                        }
                    - the [attendees] field is optional and follows the definition below
                        "attendees": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "email": {
                                        "type": "string",
                                        "format": "email"
                                    },
                                    "display_name": {
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "email"
                                ]
                            }
                        }
                    - do not assume you know the attendee's email address if none is provided, prompt the user to provide an email address
                    - all fields in the ["add_an_event"] category should be present, if no value is defaulted to a field or specified by the user, assign the field to an empty string or empty list
                    - if required fields are missing, prompt the user to provide the information
                    - only if all required fields are filled, prompt the user to confirm all required fields and add to any optional fields if none are provided
                        - for example, "please confirm, [category] to the calendar, the [event]"
                        - provide a helpful reminder for any optional [fields]s left empty
                        - upon confirmation, set the confirmation field to true
                        - until the user has confirmed the event, assume the intent is to add an event until it is confirmed 
                ["delete_an_event"]: event-id (required)
                ["update_an_event]: event-id (required), title (optional), start-time (optional), start-date (optional), end-time (optional), end-date (optional), summary (optional), notes (optional), contacts (optional), should-notify-contact (optional), set-reminder (optional) 
            
        """,
    }
]


print("hello, i'm your ai assistance, how can i help you?")

# Main loop for the chat
while True:
    try:
        # Get user input
        user_input = input("You: ")

        if user_input == "###":
            break  # Exit the chat if the user types ###
        else:
            # Append the user message to the conversation
            conversation.append({"role": "user", "content": user_input})

            try:
                completion = agent.run(conversation)
            except Exception as e:
                print(f"Error generating response: {e}")
                continue  # Skip the rest of the loop and prompt for user input again

            data = json.loads(completion)
            # listens for calendar intent
            if data["intent"] == "calendar":
                if data["add_an_event"]["confirmed"]:
                    event = data["add_an_event"]
                    completion = create_event(
                        start_time_str=f"{event['start_date']} {event['start_time']}",
                        end_time_str=f"{event['end_date']} {event['end_time']}",
                        summary=event["title"],
                        description=event["summary"],
                        location=event["title"],
                        attendees=event["attendees"],
                        reminders=event["reminders"],
                    )

            # Append the AI's response to the conversation
            conversation.append({"role": "assistant", "content": completion})

            # Print the AI's response
            print(f"Assistant: {completion}")

    except KeyboardInterrupt:
        print("\nExiting chat...")
        break  # Exit the chat if Ctrl+C is pressed
    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally, you can decide to break the loop or continue after handling the error
        # For this example, let's choose to continue
        continue
