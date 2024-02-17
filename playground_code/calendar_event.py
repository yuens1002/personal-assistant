class event:
    def __init__(self):
        self.events = {}

    def add_event(self, date_str, time_str, event_name):
        self.events[(date_str, time_str)] = event_name

    # lists all events
    def list_events(self):
        for (date_str, time_str), event_name in self.events.items():
            print(f"{event_name} on {date_str} at {time_str}")

    # if date & time available
    def is_availability(self):
        if self.event_name:
            return True
        return False
