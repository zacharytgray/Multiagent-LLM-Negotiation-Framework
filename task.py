from psrMappings import psrMapping, taskMapping

class Task:
    def __init__(self, name, pref1, pref2):
        self.name = name
        self.mappedName = taskMapping[name] # Map the task name to a more human-readable format
        self.pref1 = pref1
        self.pref2 = pref2
        self.confidence1 = psrMapping[pref1] # Map the preference to a more human-readable format
        self.confidence2 = psrMapping[pref2] # Map the preference to a more human-readable format

    def __repr__(self):
        return f"{self.mappedName} ({self.pref1}, {self.pref2})"