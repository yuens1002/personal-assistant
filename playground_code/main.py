from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle
import os.path

# Define the scopes
SCOPES = ["https://www.googleapis.com/auth/calendar"]

creds = None
# The file token.pickle stores the user's access and refresh tokens
if os.path.exists("token.pickle"):
    with open("token.pickle", "rb") as token:
        creds = pickle.load(token)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.pickle", "wb") as token:
        pickle.dump(creds, token)

service = build("calendar", "v3", credentials=creds)

# Define the event
event = {
    "summary": "Google I/O 2024",
    "location": "800 Howard St., San Francisco, CA 94103",
    "description": "A chance to hear more about Google's developer products.",
    "start": {
        "dateTime": "2024-03-01T09:00:00-07:00",
        "timeZone": "America/Los_Angeles",
    },
    "end": {
        "dateTime": "2024-03-01T17:00:00-07:00",
        "timeZone": "America/Los_Angeles",
    },
}

event = service.events().insert(calendarId="primary", body=event).execute()
print(f'Event created: {event.get("htmlLink")}')
