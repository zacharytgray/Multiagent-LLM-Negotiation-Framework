import csv
from datetime import datetime, timedelta
import os
from itertools import combinations
from proposal import Proposal
from task import Task

class scoringEngine:
    def __init__(self, logFilename):        
        self.rounds = [] # List of rounds, set in parseLog
        self.numTasks = None # Number of tasks, set in parseLog
        self.agent1Model = None # Agent 1 model, set in parseLog
        self.agent2Model = None # Agent 2 model, set in parseLog
        self.totalNegotiationTime = None # Total negotiation time, set in parseLog
        self.averageTimePerRound = None # Average time per round, set in parseLog
        
        logsFolder = "Logs"
        if not os.path.exists(logsFolder):
            os.makedirs(logsFolder)
        self.logFilepath = os.path.join(logsFolder, logFilename)
        
    def parseLog(self): 
        """
        Parse the log file
        """
        with open (self.logFilepath, mode='r', newline = '') as file:
            reader = csv.reader(file)
            header = next(reader) # Skip the header row
            for row in reader:
                if row[0] == "NumTasks":
                    self.numTasks = int(row[1])
                elif row[0] == "Agent1Model":
                    self.agent1Model = row[1]
                elif row[0] == "Agent2Model":
                    self.agent2Model = row[1]
                elif row[0] == "TotalNegotiationTime":
                    self.totalNegotiationTime = self.parseDuration(row[1])
                elif row[0] == "AverageTimePerRound":
                    self.averageTimePerRound = self.parseDuration(row[1])
                else:
                    roundNumber = int(row[0])
                    negotiationTime = self.parseDuration(row[1])
                    agent1Utility = float(row[2])
                    agent2Utility = float(row[3])
                    numIterations = int(row[4])
                    agent1Tasks = self.parseTasks(row[5])
                    agent2Tasks = self.parseTasks(row[6])
                    tasks = self.parseTasks(row[7])
                    initialProposal = self.parseProposal(row[8]) # input tuple: [agent1Tasks, agent2Tasks]
                    agent1UsesOpenAI = row[9].strip().lower() == 'true'
                    agent2UsesOpenAI = row[10].strip().lower() == 'true'
                    agent1ModelName = row[11]
                    agent2ModelName = row[12]
                    agent1Type = row[13]
                    agent2Type = row[14]
                    hasDNF = row[15].strip().lower() == 'true'
                    
                    roundData = {
                        'roundNumber': roundNumber,
                        'negotiationTime': negotiationTime,
                        'agent1Utility': agent1Utility,
                        'agent2Utility': agent2Utility,
                        'numIterations': numIterations,
                        'agent1Tasks': agent1Tasks,
                        'agent2Tasks': agent2Tasks,
                        'tasks': tasks,
                        'initialProposal': initialProposal,
                        'agent1UsesOpenAI': agent1UsesOpenAI,
                        'agent2UsesOpenAI': agent2UsesOpenAI,
                        'agent1ModelName': agent1ModelName,
                        'agent2ModelName': agent2ModelName,
                        'agent1Type': agent1Type,
                        'agent2Type': agent2Type,
                        'hasDNF': hasDNF
                    }
                    self.rounds.append(roundData)
        
    def parseTime(self, timeStr): # Parse time string into timedelta
        """
        Parse a time string into a timedelta object.
        """
        timeStr = timeStr.strip("[]")
        hours, minutes, seconds = map(float, timeStr.split(":"))
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    def parseTasks(self, tasksStr): # Parse tasks string into list of tuples
        """
        Parse a string representation of a list of tasks into a list of tuples.
        """
        tasksStr = tasksStr.strip("[]")
        tasks = tasksStr.split("), ")
        parsedTasks = []
        for task in tasks:
            name, prefs = task.split(" (")
            pref1, pref2 = prefs.strip(")").split(", ")
            parsedTasks.append(Task(name, float(pref1), float(pref2)))
        return parsedTasks    
    def parseProposal(self, proposalStr): # Parse proposal string into proposal object
        """
        Parse a string representation of a proposal into a Proposal object.
        The proposal string is expected in the following format:
        "([Task C (0.1, 0.9), Task D (0.8, 0.4), Task F (0.2, 0.6)], [Task A (0.5, 0.7), Task B (0.6, 0.2), Task E (0.5, 0.3)])"
        """
        # Remove outer parentheses if present.
        proposalStr = proposalStr.strip("()")
        
        # Split the string into two parts by the delimiter "], "
        parts = proposalStr.split("], ")
        if len(parts) != 2:
            raise ValueError("Invalid proposal format.")
        
        # Ensure the first part ends with a closing bracket.
        agent1TasksStr = parts[0].strip()
        if not agent1TasksStr.endswith("]"):
            agent1TasksStr += "]"
        
        # Ensure the second part starts with an opening bracket.
        agent2TasksStr = parts[1].strip()
        if not agent2TasksStr.startswith("["):
            agent2TasksStr = "[" + agent2TasksStr
        
        # Parse the tasks using the parseTasks method.
        agent1Tasks = self.parseTasks(agent1TasksStr)
        agent2Tasks = self.parseTasks(agent2TasksStr)
        
        return Proposal(agent1Tasks, agent2Tasks)
    def parseDuration(self, timeStr): # Parse duration string into timedelta
        """
        Parse a time string in the format H:M:S.microseconds into a timedelta object.
        """
        timeParts = timeStr.split(':')
        hours = int(timeParts[0])
        minutes = int(timeParts[1])
        seconds = float(timeParts[2])
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    def printRound(self, roundNumber): # Print round data in a human-readable format
        """
        Print the data for a given round
        """
        print(f"numTasks: {self.numTasks}")
        print(f"agent1Model: {self.agent1Model}")
        print(f"agent2Model: {self.agent2Model}")
        print(f"totalNegotiationTime: {self.totalNegotiationTime}")
        print(f"averageTimePerRound: {self.averageTimePerRound}")
        for roundData in self.rounds:
            if roundData['roundNumber'] == roundNumber:
                print(f"roundNumber: {roundData['roundNumber']}")
                print(f"negotiationTime: {roundData['negotiationTime']}")
                print(f"agent1Utility: {roundData['agent1Utility']}")
                print(f"agent2Utility: {roundData['agent2Utility']}")
                print(f"numIterations: {roundData['numIterations']}")
                print(f"agent1Tasks: {roundData['agent1Tasks']}")
                print(f"agent2Tasks: {roundData['agent2Tasks']}")
                print(f"tasks: {roundData['tasks']}")
                break          
    def getAllPossibleAllocations(self, roundTasks): # Get all possible allocations for a given round
        """
        Param: roundTasks: List of all tasks in a round
        Return a list of all possible Proposals, sorted by higher overall utility to lowest
        """
        possibleAllocations = [] 
        taskCount = len(roundTasks)
        for size in range(1, taskCount):
            for combo in combinations(roundTasks, size):
                groupA = list(combo)
                groupB = [task for task in roundTasks if task not in groupA]
                newProposal = Proposal(groupA, groupB)
                possibleAllocations.append(newProposal)
        possibleAllocations.sort(key=lambda proposal: proposal.totalUtility, reverse=True)
        return possibleAllocations
    
    def getGroupedRankedAllocations(self, roundTasks): # Get all possible allocations for a given round, where ties are grouped
        allAllocations = self.getAllPossibleAllocations(roundTasks)
        groupedRanking = {}
        
        groupIndex = 0
        previousUtility = None
        
        for allocation in allAllocations:
            if previousUtility is not None and abs(allocation.totalUtility - previousUtility) < 1e-7: # Check if the utility is very close to the previous one
                # Same utility as previous => same group
                groupedRanking[groupIndex].append(allocation)
            else:
                # New utility => increment group index, start new group
                groupIndex += 1
                groupedRanking[groupIndex] = [allocation]
                previousUtility = allocation.totalUtility

        return groupedRanking
        
    def printGroupedRankedAllocations(self, groupedRankedAllocations): # Print all possible allocations for a given round, where ties are grouped
        for groupIndex in sorted(groupedRankedAllocations.keys()):
            print(f"\nGroup {groupIndex}:")
            for proposal in groupedRankedAllocations[groupIndex]:
                print(f"Total Utility: {proposal.totalUtility}")
    
    def isOptimalAllocation(self, proposal, roundTasks): # Check if a given proposal is optimal
        return self.getAllocationRank(proposal, roundTasks) == 1

    def getAllocationRank(self, proposal, roundTasks): # Get the rank of a given allocation out of all possible allocations (1st is best)
        groupedRanking = self.getGroupedRankedAllocations(roundTasks) # all Allocations is a dictionary of index:[proposal1, proposal2, ...]
        currentGroup1 = set(proposal.agent1Tasks)
        currentGroup2 = set(proposal.agent2Tasks)
        for groupIndex in sorted(groupedRanking.keys()):
            for candidateProposal in groupedRanking[groupIndex]:
                candidateGroup1 = set(candidateProposal.agent1Tasks)
                candidateGroup2 = set(candidateProposal.agent2Tasks)
                # Check if the current proposal matches this candidate proposal
                # allowing for swapped sides (agent1 <-> agent2).
                if (currentGroup1 == candidateGroup1 and currentGroup2 == candidateGroup2) \
                or (currentGroup1 == candidateGroup2 and currentGroup2 == candidateGroup1):
                    return groupIndex
            
        return None # Not found
    
    def calculateOptimalAllocationPercentage(optimalCount, totalRounds):
        """
        Calculate the percentage of optimal allocations.
        """
        return (optimalCount / totalRounds) * 100

    def calculateAllocationScoreLoss(currentUtility, optimalUtility):
        """
        Calculate the allocation score loss as a percentage.
        """
        return 100 * (1 - (currentUtility / optimalUtility))

    def isParetoOptimal(allocation, allAllocations):
        """
        Check if the given allocation is Pareto optimal.
        """
        return True

if __name__ == "__main__":
    se = scoringEngine("gemma2_gemma2_2025-02-01_16:53:10.csv")
    se.parseLog()
    
    roundNum  = 1
    round1Tasks = se.rounds[roundNum-1]['tasks']
    round1Agent1Tasks = se.rounds[roundNum-1]['agent1Tasks']
    round1Agent2Tasks = se.rounds[roundNum-1]['agent2Tasks']
    round1Proposal = Proposal(round1Agent1Tasks, round1Agent2Tasks)
    
    allocationRank = se.getAllocationRank(round1Proposal, round1Tasks)
    print(f"Allocation rank: {allocationRank}")