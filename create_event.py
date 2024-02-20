from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle
import os.path
from datetime import datetime

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def authenticate_google_calendar():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens.
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return creds


def to_iso_time(date_time_str):
    return datetime.strptime(date_time_str, "%m/%d/%Y %I%p").isoformat()


def create_event(
    start_time_str, end_time_str, summary, description=None, location=None
):
    creds = authenticate_google_calendar()
    service = build("calendar", "v3", credentials=creds)

    event = {
        "summary": summary,
        "location": location,
        "description": description,
        "start": {
            "dateTime": to_iso_time(start_time_str),
            "timeZone": "America/New_York",
        },
        "end": {
            "dateTime": to_iso_time(end_time_str),
            "timeZone": "America/New_York",
        },
    }

    event = service.events().insert(calendarId="primary", body=event).execute()
    return f"Event created: {event.get('htmlLink')}"
