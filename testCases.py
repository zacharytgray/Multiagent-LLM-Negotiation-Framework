import unittest
import random
from colorama import Fore
import batchTaskAllocator as bta
import time
import itertools
from datetime import datetime

def main():
    numRounds = 100 # Number of rounds to be run
    numTasks = 4 # Number of tasks to be assigned per round
    numIterations = 4 # Number of conversation iterations per round
    allocationScoreCeiling = 15 # The maximum percentage away from the optimal allocation that is considered passing

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logFilename = f"log_{timestamp}.txt"

    f = open(logFilename, "w")
    f.write("TASK ALLOCATION LOG\n\n")
    f.write(f"Number of Tasks to Allocate: {numTasks}\n")
    f.write(f"Number of Rounds: {numRounds}\n")
    f.write(f"Default Number of Conversation Iterations Per Round: {numIterations}\n")
    f.write(f"Allocation Score Ceiling: Allocations must be less than {allocationScoreCeiling}% away from the optimal solution\n")
    f.close()
    add_test_methods(numRounds, numTasks, numIterations, allocationScoreCeiling, logFilename)

class TestAgent(unittest.TestCase):

    def getOptimalAllocation(self,tasks):
        optimalSolution = []
        bestPSRSum = 0
        n = len(tasks)
        half_n = n // 2

        # Ensure there are an even number of tasks
        if n % 2 != 0:
            raise ValueError("The number of tasks must be even to form two equal groups.")
        
        # Generate all combinations of half2_n tasks
        all_combinations = itertools.combinations(tasks, half_n)
                
        for comb in all_combinations:
            group1 = comb
            group2 = tuple(task for task in tasks if task not in group1)
            
            # Calculate the sum of the first PSR for group1 and the second PSR for group2
            sum1 = sum(task[1] for task in group1)
            sum2 = sum(task[2] for task in group2)
            PSR_Sum = sum1 + sum2

            if PSR_Sum > bestPSRSum:
                bestPSRSum = PSR_Sum
                optimalSolution = [group1, group2]

        return optimalSolution[0], optimalSolution[1], bestPSRSum
    
    def calculatePSR(self, agent1Tasks, agent2Tasks):
        PSR1 = sum(task[1] for task in agent1Tasks)
        PSR2 = sum(task[2] for task in agent2Tasks)
        return PSR1 + PSR2

    def hasOptimalAllocation(self, agent1Tasks, agent2Tasks, bestPSR):
        currPSR = self.calculatePSR(agent1Tasks, agent2Tasks)
        return currPSR >= bestPSR

    def getAllocationScore(self, agent1Tasks, agent2Tasks, bestPSR): # The closer to 0, the closer to the optimal allocation
        currPSR = self.calculatePSR(agent1Tasks, agent2Tasks)
        return round(100 * ((bestPSR - currPSR) / bestPSR))

    def setUp(self, numTasks):
        self.numTasks = numTasks  # number of tasks to be assigned

    def run_round(self, round_num, numRounds, numIterations, allocationScoreCeiling, logFilename):
        allocationErrorFound = False

        with open(logFilename, "a") as f:
            f.write(("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")
            f.close()
        print("\n" + ("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")

        # Generate random tasks and PSRs
        agent1 = bta.Agent("Agent 1")
        agent2 = bta.Agent("Agent 2")
        tasks = []  # formatted as [('Task X', PSR1, PSR2), ...]
        taskNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
        for i in range(self.numTasks):  # Generate random tasks
            task = f"Task {taskNames[i]}"
            PSR1 = round(random.uniform(0, 1), 1) # Generate random PSRs between 0 and 1, rounded to 1 decimal place
            PSR2 = round(random.uniform(0, 1), 1)
            tasks.append((task, PSR1, PSR2))

        domain = bta.Domain(agent1, agent2, tasks)
        domain.assignTasks(numIterations)
        agent1Tasks = domain.agent1.assignedTasks  # formatted as [('Task X', PSR1, PSR2), ...]
        agent2Tasks = domain.agent2.assignedTasks
        domain.printTasks()

        # bta.logMemoryBuffer(self.fileName, agent1, agent2)  # Log memory buffers
        f = open(logFilename, "a")
        f.write("\nTasks for this round:\n")
        for task in tasks:
            f.write(f"- {task}\n")
        f.close()
        bta.logAssignedTasks(logFilename, agent1, agent2)  # Log assigned tasks

        f = open(logFilename, "a")
        optimalAllocation1, optimalAllocation2, bestPSR = self.getOptimalAllocation(tasks)
        allocationScore = self.getAllocationScore(agent1Tasks, agent2Tasks, bestPSR)
        if allocationScore < allocationScoreCeiling:
            f.write(f"\nAllocation Score (PASSING): {allocationScore}% away from the optimal.\n")
            print(f"\n{Fore.GREEN}Allocation Score (PASSING): {allocationScore}% away from the optimal.{Fore.RESET}")
        else:
            f.write(f"\nAllocation Score (FAILING): {allocationScore}% away from the optimal.\n")
            print(f"\n{Fore.RED}Allocation Score (FAILING): {allocationScore}% away from the optimal.{Fore.RESET}")
            allocationErrorFound = True
        hasOptimalAllocation = self.hasOptimalAllocation(agent1Tasks, agent2Tasks, bestPSR)
        if not hasOptimalAllocation:
            
            f.write("Optimal Allocation:\n")
            f.write(f"  Agent 1: {optimalAllocation1}\n")
            f.write(f"  Agent 2: {optimalAllocation2}\n")
            
            print("\nOptimal Allocation:\n")
            print(f"    Agent 1: {optimalAllocation1}")
            print(f"    Agent 2: {optimalAllocation2}")
            
        f.write(f"\nNumber of conversation iterations: {domain.numConversationIterations}")
        f.write(f"\nNumber of tokens generated by {agent1.name}: {agent1.numTokensGenerated}")
        f.write(f"\nNumber of tokens generated by {agent2.name}: {agent2.numTokensGenerated}\n")
        f.close()
        
        return hasOptimalAllocation, allocationErrorFound, allocationScore
    
def format_seconds(seconds):
    # Ensure seconds is an integer
    total_seconds = int(seconds)
    
    # Calculate hours, minutes, and remaining seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    
    # Format into "hh:mm:ss" with zero-padding
    formatted_time = f"{hours:02}:{minutes:02}:{remaining_seconds:02}"
    
    return formatted_time

def add_test_methods(numRounds, numTasks, numIterations, allocationScoreCeiling, logFilename):
    numOptimal = 0
    numPassing = 0
    totalAllocationScore = 0
    ta = TestAgent()
    ta.setUp(numTasks)
    totalTime = 0

    f = open(logFilename, "a")
    agent = bta.Agent("Dummy Agent")
    f.write(f"LLM Being Used to Allocate Tasks: {agent.model}\n")
    f.close()

    for i in range(numRounds):
        startTime = time.time()
        hasOptimalAllocation, allocationErrorFound, allocationScore = ta.run_round(i+1, numRounds, numIterations, allocationScoreCeiling, logFilename)
        totalAllocationScore += allocationScore
        endTime = time.time()
        duration = endTime - startTime
        totalTime += duration
        numOptimal += 1 if hasOptimalAllocation else 0
        numPassing += 1 if not allocationErrorFound else 0
        with open (logFilename, "a") as f:
            f.write(f"Round {i+1} Duration: Completed in {format_seconds(duration)}\n\n")
            f.close()

    with open(logFilename, "a") as f:
        f.write(("=" * 25) + f"  TOTAL  " + ("=" * 25) + "\n")
        f.write(f"\nTotal Optimal Allocations: {numOptimal} of {numRounds} rounds were optimal.")
        f.write(f"\nTotal Passing Allocations: {numPassing} of {numRounds} rounds were within the allocation test tolerance.")
        f.write(f"\nAverage Allocation Score: {round(totalAllocationScore/numRounds)}% away from the optimal.")
        f.write(f"\nTotal Time: {format_seconds(totalTime)}")
        f.write(f"\nAverage Time per Round: {format_seconds(totalTime/numRounds)}\n")
        f.close()

main()