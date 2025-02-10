import csv
from datetime import datetime, timedelta
import os
from itertools import combinations
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
from proposal import Proposal
from task import Task

#TODO: Need to implement:

class scoringEngine:
    def __init__(self, logFilename):        
        self.rounds = [] # List of rounds, set in parseLog
        self.numTasks = None # Number of tasks, set in parseLog
        self.agent1Model = None # Agent 1 model, set in parseLog
        self.agent2Model = None # Agent 2 model, set in parseLog
        self.totalNegotiationTime = None # Total negotiation time, set in parseLog
        self.averageTimePerRound = None # Average time per round, set in parseLog
        
        # Allocation Tolerance (in %): A percentage threshold for all allocation rounds. 
            # If the Allocation Score Loss is less than or equal to this threshold, the allocation passes this test. 
            # If the Allocation Score Loss is greater, it fails. 
        self.allocationTolerance = 0.15 # Allocation tolerance
        
        logsFolder = "Logs"
        if not os.path.exists(logsFolder):
            os.makedirs(logsFolder)
        self.logFilepath = os.path.join(logsFolder, logFilename)
        
    def parseLog(self): 
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
                    initialProposal = self.parseProposal(row[8]) # input tuple: [agent1Tasks, agent2Tasks] or None if hasInitialProposal == False
                    agent1UsesOpenAI = row[9].strip().lower() == 'true'
                    agent2UsesOpenAI = row[10].strip().lower() == 'true'
                    agent1ModelName = row[11]
                    agent2ModelName = row[12]
                    agent1Type = row[13]
                    agent2Type = row[14]
                    
                    winningProposal = Proposal(agent1Tasks, agent2Tasks, True)
                    
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
                        'winningProposal': winningProposal
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
        
        # Handle empty list case
        if tasksStr.strip() == "[]":
            return []
        
        tasksStr = tasksStr.strip("[]")
        tasks = tasksStr.split("), ")
        parsedTasks = []
        for task in tasks:
            name, prefs = task.split(" (")
            pref1, pref2 = prefs.strip(")").split(", ")
            parsedTasks.append(Task(name, float(pref1), float(pref2)))
        return parsedTasks    
    def parseProposal(self, proposalStr): # Parse proposal string into proposal object 
        if proposalStr == "": # Handle the case where there is no initial proposal
            return None
            
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
        for size in range(0, taskCount + 1):
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
    
    def calculateOptimalAllocationPercentage(self, optimalCount, totalRounds):
        """
        Calculate the percentage of optimal allocations.
        """
        return (optimalCount / totalRounds) * 100

    def getAllocationScoreLoss(self, currentUtility, optimalUtility):
        # For a given allocation, this is the percentage drop from the optimal allocation. 
        # It measures the difference from the optimal allocation as a percentage of that maximum. 
        # This is calculated as follows:
        # 𝐿𝑜𝑠𝑠 = 100 ∗ (1 − (Current Task Utility/Optimal Task Utility))
        """
        Calculate the allocation score loss as a percentage.
        """
        return 100 * (1 - (currentUtility / optimalUtility))
    
    def getOptimalAllocationPercentage(self):
        # Returns the percentage of negotiations in which the agents achieved the optimal allocation. 
        """
        Calculate the percentage of optimal allocations.
        """
        optimalCount = sum(1 for roundData in self.rounds if self.isOptimalAllocation(roundData['winningProposal'], roundData['tasks']))
        return self.calculateOptimalAllocationPercentage(optimalCount, len(self.rounds))


    def getPercentageAwayFromOptimal(self, agent1Items, agent2Items):
        # Create proposal from the given items
        currentProposal = Proposal(agent1Items, agent2Items)
        currentUtility = currentProposal.totalUtility
        
        # Get all possible allocations for these tasks
        allTasks = agent1Items + agent2Items
        allAllocations = self.getAllPossibleAllocations(allTasks)
        
        # Get the best possible utility (first allocation since list is sorted)
        bestUtility = allAllocations[0].totalUtility if allAllocations else 0
        
        # Calculate percentage difference
        percentAway = round(100 * abs(currentUtility - bestUtility) / bestUtility, 2)
        return percentAway
        
    def getPercentageWithinAllocationTolerance(self):
        # Percentage within Allocation Tolerance (in %): 
        # The percentage of rounds whose allocations were within the allocation tolerance. 
        # That is, the percentage of all rounds that are close enough to the optimal solution to be considered passing.
        
        numWithinTolerance = 0
        totalRounds = len(self.rounds)
        
        for roundData in self.rounds:
            percentageAway = self.getPercentageAwayFromOptimal(roundData['agent1Tasks'], roundData['agent2Tasks'])
            if percentageAway <= self.allocationTolerance * 100:  # Convert tolerance to percentage
                numWithinTolerance += 1
                
        return (numWithinTolerance / totalRounds) * 100 if totalRounds > 0 else 0
    
    def exportUtilityComparison(self):
        """
        Create a CSV file comparing current total utility vs optimal utility for each round
        Filename format: utilityComparison-agent1model_agent2model.csv
        Columns: Round Number, Current Utility, Optimal Utility
        """
        # Sanitize model names for filename
        agent1_name = ''.join(filter(str.isalnum, self.agent1Model))
        agent2_name = ''.join(filter(str.isalnum, self.agent2Model))
        outputFilename = f"utilityComparison-{agent1_name}_{agent2_name}.csv"
        outputPath = os.path.join("Logs", outputFilename)
        
        with open(outputPath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Round', 'Current Utility', 'Optimal Utility'])  # Updated header
            
            for round_data in self.rounds:
                # Get round number
                round_num = round_data['roundNumber']
                
                # Calculate current total utility
                current_utility = round_data['agent1Utility'] + round_data['agent2Utility']
                
                # Get optimal utility for this round
                all_allocations = self.getAllPossibleAllocations(round_data['tasks'])
                optimal_utility = all_allocations[0].totalUtility if all_allocations else 0
                
                writer.writerow([round_num, current_utility, optimal_utility])
    
    def createUtilityComparisonPlot(self):
        """
        Create and display a plot comparing current total utility vs optimal utility for each round
        """
        
        rounds = []
        current_utilities = []
        optimal_utilities = []
        
        for round_data in self.rounds:
            # Get round number and current utility
            rounds.append(round_data['roundNumber'])
            current_utilities.append(round_data['agent1Utility'] + round_data['agent2Utility'])
            
            # Get optimal utility for this round
            all_allocations = self.getAllPossibleAllocations(round_data['tasks'])
            optimal_utilities.append(all_allocations[0].totalUtility if all_allocations else 0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, optimal_utilities, 'b-', label='Optimal Utility', marker='o')
        plt.plot(rounds, current_utilities, 'r-', label='Current Utility', marker='o')
        
        plt.xlabel('Round Number')
        plt.ylabel('Utility')
        plt.title(f'Utility Comparison by Round\n{self.agent1Model} vs {self.agent2Model}')
        plt.legend()
        plt.grid(True)
        
        plt.show()

if __name__ == "__main__":
    se = scoringEngine("llama33latest_llama33latest_2025-02-09_15:47:12.csv")
    se.parseLog()
    
    numRounds = len(se.rounds)
    numOptimal = 0
    allocationRankSum = 0
    
    for roundData in se.rounds:
        roundNum = roundData['roundNumber']
        roundTasks = se.rounds[roundNum-1]['tasks']
        roundAgent1Tasks = se.rounds[roundNum-1]['agent1Tasks']
        roundAgent2Tasks = se.rounds[roundNum-1]['agent2Tasks']
        roundProposal = Proposal(roundAgent1Tasks, roundAgent2Tasks)
        allocationRank = se.getAllocationRank(roundProposal, roundTasks)
        
        if se.isOptimalAllocation(roundProposal, roundTasks):
            numOptimal += 1
        
        allocationRankSum += allocationRank
            
            
    print(f"Average Allocation Rank: {allocationRankSum/numRounds}")    
    print(f"Average Allocation Score Loss: {se.getAllocationScoreLoss(roundProposal.totalUtility, se.getAllPossibleAllocations(roundTasks)[0].totalUtility)}%")
    print(f"Optimal allocation percentage: {se.getOptimalAllocationPercentage()}%")
    print(f"Percentage within allocation tolerance: {se.getPercentageWithinAllocationTolerance()}%")
    # se.exportUtilityComparison()
    se.createUtilityComparisonPlot()
