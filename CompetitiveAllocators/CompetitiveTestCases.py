import unittest
import random
from colorama import Fore
import CompetitiveTaskAllocator as compTA
import time
import itertools
from datetime import datetime

def main():
    numRounds = 10 # Number of rounds to be run
    numitems = 4 # Number of items to be assigned per round
    numIterations = 4 # Number of conversation iterations per round
    allocationScoreCeiling = 15 # The maximum percentage away from the optimal allocation that is considered passing
    model = 'llama3.1:8b' # The LLM model to be used for allocation
    
    print("\n" + ("=" * 25) + f"  COMPETITIVE ITEM ALLOCATION TEST  " + ("=" * 25) + "\n")
    print(f"Number of Items to Allocate: {numitems}")
    print(f"Number of Rounds: {numRounds}")
    print(f"Default Number of Conversation Iterations Per Round: {numIterations}")
    print(f"Allocation Score Ceiling: Allocations must be less than {allocationScoreCeiling}% away from the optimal solution")
    print(f"LLM Being Used to Allocate items: {model}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logFilename = f"log_CompTA_{timestamp}.txt"

    f = open(logFilename, "w")
    f.write("COMPETITIVE ITEM ALLOCATION LOG\n\n")
    f.write(f"Number of Items to Allocate: {numitems}\n")
    f.write(f"Number of Rounds: {numRounds}\n")
    f.write(f"Default Number of Conversation Iterations Per Round: {numIterations}\n")
    f.write(f"Allocation Score Ceiling: Allocations must be less than {allocationScoreCeiling}% away from the optimal solution\n")
    f.write(f"LLM Being Used to Allocate items: {model}\n")
    f.close()
    add_test_methods(numRounds, numitems, numIterations, allocationScoreCeiling, logFilename, model)
    
    
    # ta = TestAgent()
    # ta.setUp(4)
    # items = [compTA.Item("A", 1, 2), compTA.Item("B", 2, 1), compTA.Item("C", 3, 3), compTA.Item("D", 4, 4)]
    # print("Optimal: " + str(ta.getOptimalAllocation(items)))
    # print("Has Optimal: " + str(ta.hasOptimalAllocation([items[0], items[2]], [items[1], items[3]], 9)))
    # print("Allocation Score: " + str(ta.getAllocationScore([items[0], items[2]], [items[1], items[3]], 7)))

class TestAgent(unittest.TestCase):
    
    def getOptimalAllocation(self, items: list[compTA.Item]):
        optimalSolution = []
        lowestPrefSum = float('inf')
        n = len(items)
        half_n = n // 2

        # Ensure there are an even number of items
        if n % 2 != 0:
            raise ValueError("The number of items must be even to form two equal groups.")
        
        # Generate all combinations of half2_n items
        all_combinations = itertools.combinations(items, half_n)
                
        for comb in all_combinations:
            group1 = comb
            group2 = tuple(item for item in items if item not in group1)
            
            # Calculate the sum of the first prefSum for group1 and the second prefSum for group2
            sum1 = sum(item.pref1 for item in group1)
            sum2 = sum(item.pref2 for item in group2)
            PrefSum = sum1 + sum2
            
            if PrefSum < lowestPrefSum:
                lowestPrefSum = PrefSum
                optimalSolution = [group1, group2]

        return optimalSolution[0], optimalSolution[1], lowestPrefSum
    
    def calculatePrefSum(self, agent1items, agent2items):
        p1 = sum(item.pref1 for item in agent1items)
        p2 = sum(item.pref2 for item in agent2items)
        return p1 + p2

    def hasOptimalAllocation(self, agent1items, agent2items, bestPrefSum):
        currPrefSum = self.calculatePrefSum(agent1items, agent2items)
        return currPrefSum <= bestPrefSum

    def getAllocationScore(self, agent1items, agent2items, bestprefSum): # The closer to 0, the closer to the optimal allocation
        currprefSum = self.calculatePrefSum(agent1items, agent2items)
        return round(100 * ((abs(currprefSum - bestprefSum)) / bestprefSum))

    def setUp(self, numItems):
        self.numItems = numItems  # number of items to be assigned

    def run_round(self, round_num, numRounds, numIterations, allocationScoreCeiling, logFilename, model):
        allocationErrorFound = False

        print("\n" + ("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")
        with open(logFilename, "a") as f:
            f.write("\n" + ("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")
            f.close()

        # Generate random items and prefSums
        items = [] # list of Item objects
        itemNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
        prefOrder1 = self.generateRandomPrefOrder()
        prefOrder2 = self.generateRandomPrefOrder()
            
        for i in range(self.numItems):  # Generate random items
            item = f"Item {itemNames[i]}"
            pref1 = prefOrder1[i]
            pref2 = prefOrder2[i]
            items.append(compTA.Item(item, pref1, pref2))
            
        domain = compTA.Domain(items, model)
        domain.startNegotiation(numIterations)
        agent1items = domain.boardState.getItems(domain.agent1.name)  # formatted as [('item X', prefSum1, prefSum2), ...]
        agent2items = domain.boardState.getItems(domain.agent2.name)
        
        domain.printItems()

        f = open(logFilename, "a")
        f.write("\nItems for this round:\n")
        for item in items:
            f.write(f"- {item}\n")
        f.close()
        compTA.logAssignedItems(logFilename, domain.boardState)  # Log assigned items

        f = open(logFilename, "a")
        optimalAllocation1, optimalAllocation2, bestprefSum = self.getOptimalAllocation(items)

        allocationScore = self.getAllocationScore(agent1items, agent2items, bestprefSum)
        if allocationScore < allocationScoreCeiling:
            f.write(f"\nAllocation Score (PASSING): {allocationScore}% away from the optimal.\n")
            print(f"\n{Fore.GREEN}Allocation Score (PASSING): {allocationScore}% away from the optimal.{Fore.RESET}")
        else:
            f.write(f"\nAllocation Score (FAILING): {allocationScore}% away from the optimal.\n")
            print(f"\n{Fore.RED}Allocation Score (FAILING): {allocationScore}% away from the optimal.{Fore.RESET}")
            allocationErrorFound = True
        hasOptimalAllocation = self.hasOptimalAllocation(agent1items, agent2items, bestprefSum)
        if not hasOptimalAllocation:
            
            f.write("Optimal Allocation:\n")
            f.write(f"  Agent 1: {optimalAllocation1}\n")
            f.write(f"  Agent 2: {optimalAllocation2}\n")
            
            print("\nOptimal Allocation:\n")
            print(f"    Agent 1: {optimalAllocation1}")
            print(f"    Agent 2: {optimalAllocation2}")
            
        f.write(f"\nNumber of conversation iterations: {domain.numConversationIterations}")
        f.write(f"\nNumber of tokens generated by {domain.agent1.name}: {domain.agent1.numTokensGenerated}")
        f.write(f"\nNumber of tokens generated by {domain.agent2.name}: {domain.agent2.numTokensGenerated}\n")
        f.close()
        
        return hasOptimalAllocation, allocationErrorFound, allocationScore
    
    def generateRandomPrefOrder(self):
        stack = []
        for i in range(1, self.numItems + 1):
            stack.append(i)
        random.shuffle(stack)
        return stack
    
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

def add_test_methods(numRounds, numitems, numIterations, allocationScoreCeiling, logFilename, model):
    numOptimal = 0
    numPassing = 0
    totalAllocationScore = 0
    ta = TestAgent()
    ta.setUp(numitems)
    totalTime = 0

    for i in range(numRounds):
        startTime = time.time()
        hasOptimalAllocation, allocationErrorFound, allocationScore = ta.run_round(i+1, numRounds, numIterations, allocationScoreCeiling, logFilename, model)
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