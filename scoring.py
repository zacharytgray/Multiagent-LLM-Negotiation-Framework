import csv
from datetime import datetime, timedelta
import os
from itertools import combinations
from proposal import Proposal
from task import Task

class scoringEngine:
    def __init__(self, logFilename):        
        self.rounds = []
        self.currentRound = None
        self.numTasks = None
        self.agent1Model = None
        self.agent2Model = None
        self.totalNegotiationTime = None
        
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
                    
                    roundData = {
                        'roundNumber': roundNumber,
                        'negotiationTime': negotiationTime,
                        'agent1Utility': agent1Utility,
                        'agent2Utility': agent2Utility,
                        'numIterations': numIterations,
                        'agent1Tasks': agent1Tasks,
                        'agent2Tasks': agent2Tasks,
                        'tasks': tasks
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
            
    def getAllPossibleAllocations(self, roundTasks):
        """
        Param: roundItems: List of all items in a round
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
    
def isOptimalAllocation(allocation, allPossibleAllocations):
    """
    Check if the given allocation is optimal.
    """
    return True

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

se = scoringEngine("deepseekr132b_deepseekr132b_2025-01-31_17:37:46.csv")
se.parseLog()
round1Tasks = se.rounds[0]['tasks']
allAllocations = se.getAllPossibleAllocations(round1Tasks)
print(allAllocations[0])
print(allAllocations[len(allAllocations) - 1])