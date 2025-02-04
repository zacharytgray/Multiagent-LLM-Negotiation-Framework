from psrMappings import psrMapping, taskMapping

class Task:
    def __init__(self, name, pref1, pref2):
        self.name = name
        try:
            self.mappedName = taskMapping[name] # Map the task name to a more human-readable format
        except KeyError:
            print(f"Warning: {name} not found in taskMapping")
            self.mappedName = name
        self.pref1 = pref1
        self.pref2 = pref2
        self.confidence1 = psrMapping[pref1] # Map the preference to a more human-readable format
        self.confidence2 = psrMapping[pref2] # Map the preference to a more human-readable format
        
    def __eq__(self, other):
        if isinstance(other, Task):
            return self.name == other.name and abs(self.pref1 - other.pref1) < 1e-7 and abs(self.pref2 - other.pref2) < 1e-7
        return False
    
    def __hash__(self):
        return hash((self.name, round(self.pref1, 7), round(self.pref2, 7)))
    
    # def getItemString(self):
    #     return f"{self.mappedName} ({self.pref1}, {self.pref2})"

    # def __repr__(self):
    #     return f"{self.name} ({self.pref1}, {self.pref2})"
    
    def __repr__(self):
        return f"{self.mappedName} ({self.pref1}, {self.pref2})"