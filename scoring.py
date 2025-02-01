import csv
from datetime import datetime, timedelta
import os
from itertools import combinations
from proposal import Proposal
from task import Task

class scoringEngine:
    def __init__(self, logFilename):        
        self.rounds = [] # List of rounds, set in parseLog
        # Each round is a dict with keys: 
            # {roundNumber, 
            # negotiationTime, 
            # agent1Utility, 
            # agent2Utility, 
            # numIterations, 
            # agent1Tasks, 
            # agent2Tasks, 
            # tasks}
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
    def getAllPossibleAllocations(self, roundTasks): # Get all possible allocations for a given round
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
    
    def getGroupedRankedAllocations(self, roundItems): # Get all possible allocations for a given round, where ties are grouped
        allAllocations = self.getAllPossibleAllocations(roundItems)
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
    
    def isOptimalAllocation(self, proposal, roundItems): # Check if a given proposal is optimal
        return self.getAllocationRank(proposal, roundItems) == 1

    def getAllocationRank(self, proposal, roundItems): # Get the rank of a given allocation out of all possible allocations (1st is best)
        groupedRanking = self.getGroupedRankedAllocations(roundItems) # all Allocations is a dictionary of index:[proposal1, proposal2, ...]
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
    se = scoringEngine("gemma2_gemma2_2025-01-31_21:21:15.csv")
    se.parseLog()
    
    roundNum  = 5
    round1Tasks = se.rounds[roundNum-1]['tasks']
    round1Agent1Tasks = se.rounds[roundNum-1]['agent1Tasks']
    round1Agent2Tasks = se.rounds[roundNum-1]['agent2Tasks']
    round1Proposal = Proposal(round1Agent1Tasks, round1Agent2Tasks)
    
    allocationRank = se.getAllocationRank(round1Proposal, round1Tasks)
    print(f"Allocation rank: {allocationRank}")