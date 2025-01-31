from negotiationFlag import NegotiationFlag

class Proposal:
    def __init__(self, agent1Tasks, agent2Tasks):
        self.agent1Tasks = agent1Tasks
        self.agent2Tasks = agent2Tasks
        self.numTasks = len(agent1Tasks) + len(agent2Tasks)
        self.agent1Utility = sum([task.pref1 for task in agent1Tasks])
        self.agent2Utility = sum([task.pref2 for task in agent2Tasks])
            
    def validateProposal(self, tasks): # Check if the proposal is valid (returns True if valid)
        if self.numTasks > len(tasks):
            return NegotiationFlag.TOO_MANY_ITEMS
        if self.numTasks < len(tasks):
            return NegotiationFlag.NOT_ENOUGH_ITEMS
        for task in tasks:
            if task not in self.agent1Tasks and task not in self.agent2Tasks: # Check if all tasks are in the proposal
                return NegotiationFlag.INVALID_ITEMS_PRESENT
        return NegotiationFlag.ERROR_FREE
    
    def __repr__(self):
        return f"Agent 1 Tasks: {self.agent1Tasks}\nAgent 2 Tasks: {self.agent2Tasks}"