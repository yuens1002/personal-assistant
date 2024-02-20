from extract_event import nlp_understand_intent, nlp_extract_details, process_to_doc
from create_event import create_event


def handle_calendar_request(user_input, event_desc):
    # Step 1: Understand User Intent
    intent = nlp_understand_intent(user_input)
    print("from handle_calendar_request/intent: ", intent)
    # Step 2: Extract Event Details
    event_details = nlp_extract_details(user_input, event_desc=event_desc)
    print("from handle_calendar_request/event_details: ", event_details)
    start = f"{event_details['dates'][0]} {event_details['times'][0]}"
    end = f"{event_details['dates'][0]} {event_details['times'][0]}"

    if intent == "create_event":
        # Step 3: Use Google Calendar API to create the event
        create_event(
            start_time_str=start,
            end_time_str=end,
            summary=event_details["summary"],
            description=event_details["description"],
            location=event_details["locations"],
        )
        response = "Your event has been scheduled."
    # elif intent == "query_events":
    #     events = query_google_calendar_events(event_details)
    #     response = format_events_for_display(events)
    else:
        response = "I'm not sure how to help with that."

    # Step 4: Communicate the outcome
    print(response)
