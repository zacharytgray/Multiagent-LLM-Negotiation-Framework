import unittest
import random
from colorama import Fore
import CompetitiveTaskAllocator as compTA
import time
import itertools
from datetime import datetime

def main():
    numRounds = 5 # Number of rounds to be run
    numitems = 4 # Number of items to be assigned per round
    numIterations = 4 # Number of conversation iterations per round
    distanceFromOptimalCeiling = 15 # The maximum percentage away from the optimal allocation that is considered passing
    model = 'gpt-4o-mini' # The LLM model to be used for allocation
    useOpenAI = True # Toggle to switch between local LLM and OpenAI LLM
    
    print("\n" + ("=" * 25) + f"  COMPETITIVE ITEM ALLOCATION TEST  " + ("=" * 25) + "\n")
    print(f"Number of Items to Allocate: {numitems}")
    print(f"Number of Rounds: {numRounds}")
    print(f"Default Number of Conversation Iterations Per Round: {numIterations}")
    print(f"Total Distance from Optimal Ceiling: Allocations must be less than {distanceFromOptimalCeiling}% away from the optimal solution")
    print(f"LLM Being Used to Allocate items: {model}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logFilename = f"log_CompTA_{timestamp}.txt"

    f = open(logFilename, "w")
    f.write("COMPETITIVE ITEM ALLOCATION LOG\n\n")
    f.write(f"Number of Items to Allocate: {numitems}\n")
    f.write(f"Number of Rounds: {numRounds}\n")
    f.write(f"Default Number of Conversation Iterations Per Round: {numIterations}\n")
    f.write(f"Total Distance from Optimal Ceiling: Allocations must be less than {distanceFromOptimalCeiling}% away from the optimal solution\n")
    f.write(f"LLM Being Used to Allocate items: {model}\n")
    f.close()
    add_test_methods(numRounds, numitems, numIterations, distanceFromOptimalCeiling, logFilename, model, useOpenAI)

class TestAgent(unittest.TestCase):
        
    def getOptimalAllocation(self, items: list[compTA.Item]): # Does not have even distribution constraint
        optimalSolution = []
        bestPrefSum = 0
        n = len(items)

        # Ensure there are an even number of items
        if n % 2 != 0:
            raise ValueError("The number of items must be even to form two equal groups.")
        
        # Generate all combinations of n items
        all_combinations = []
        for r in range(0, n):
            all_combinations.extend(itertools.combinations(items, r))


        for comb in all_combinations:
            group1 = comb
            group2 = tuple(item for item in items if item not in group1)
            
            PrefSum = self.calculatePrefSum(group1, group2)
            
            if PrefSum > bestPrefSum:
                bestPrefSum = PrefSum
                optimalSolution = [group1, group2]

        return optimalSolution[0], optimalSolution[1], bestPrefSum
       
    def isParetoOptimal(self, domain):
        agent1 = domain.agent1
        agent2 = domain.agent2
        agent1items = domain.boardState.getItems(agent1.name)
        agent2items = domain.boardState.getItems(agent2.name)
        items = domain.items
        currentAllocation = (agent1items, agent2items)
        agent1Score = sum(item.pref1 for item in agent1items)
        agent2Score = sum(item.pref2 for item in agent2items)
        
        # Get all combinations to create the allocations
        allCombinations = []
        for r in range(0, len(items)):
            allCombinations.extend(itertools.combinations(items, r))        
        
        # Get all allocations
        allAllocations = []
        for comb in allCombinations:
            group1 = comb
            group2 = tuple(item for item in items if item not in group1)
            allAllocations.append((group1, group2))

        # Remove the current allocation from all allocations
        allAllocations = [
            allocation for allocation in allAllocations
            if not (set(allocation[0]) == set(currentAllocation[0]) and set(allocation[1]) == set(currentAllocation[1]))
        ]
        
        currentScore1 = sum(item.pref1 for item in currentAllocation[0])
        currentScore2 = sum(item.pref2 for item in currentAllocation[1])
        
        for allocation in allAllocations:
            score1 = sum(item.pref1 for item in allocation[0])
            score2 = sum(item.pref2 for item in allocation[1])
            # (If new allocation can improve the first score without hurting the second score) OR (If new allocation can improve the second score without hurting the first score), then it dominates the current allocation and the current allocation is not Pareto Optimal.
            if (score1 > currentScore1 and score2 >= currentScore2) or (score2 > currentScore2 and score1 >= currentScore1):
                
                print(f"\nNot Pareto Optimal.\n")
                print("Current Allocation:\n ")
                print("Agent 1: ", currentAllocation[0])
                print("Agent 2: ", currentAllocation[1])
                print("\nWinning Allocation:\n ")
                print("Agent 1: ", allocation[0])
                print("Agent 2: ", allocation[1])
                
                return False, agent1Score, agent2Score
        
        return True, agent1Score, agent2Score
    
    def calculatePrefSum(self, agent1items, agent2items):
        p1 = sum(item.pref1 for item in agent1items)
        p2 = sum(item.pref2 for item in agent2items)
        return p1 + p2

    def hasOptimalAllocation(self, agent1items, agent2items, bestPrefSum):
        currPrefSum = self.calculatePrefSum(agent1items, agent2items)
        return currPrefSum >= bestPrefSum
    
    def getTotalDistanceFromOptimal(self, agent1items, agent2items, bestprefSum): # The closer to 0, the closer to the optimal allocation
        currprefSum = self.calculatePrefSum(agent1items, agent2items)
        return round(100 * ((abs(currprefSum - bestprefSum)) / bestprefSum))


    def setUp(self, numItems):
        self.numItems = numItems  # number of items to be assigned

    def run_round(self, round_num, numRounds, numIterations, distanceFromOptimalCeiling, logFilename, model, useOpenAI):
        exceededDistFromOptimal = False

        print("\n" + ("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")
        with open(logFilename, "a") as f:
            f.write("\n" + ("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")
            f.close()

        # Generate items and prefSums
        items = [] # list of Item objects
        itemNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
        
        # Asymmetric Preference Values - easiest case
        # prefValues1 = [0.1, 0.3, 0.6, 1.0]
        # prefValues2 = [1.0, 0.6, 0.3, 0.1]
        
        # Random Preference Values
        prefValues1 = self.generateRandomPrefValues()
        prefValues2 = self.generateRandomPrefValues()
        
        for i in range(self.numItems):  # Generate items
            item = f"Item {itemNames[i]}"
            pref1 = prefValues1[i]
            pref2 = prefValues2[i]
            items.append(compTA.Item(item, pref1, pref2))
            
        domain = compTA.Domain(items, model, useOpenAI)

        #Alternate starting agent
        startingAgent = domain.agent1 if round_num % 2 == 1 else domain.agent2
        domain.startNegotiation(numIterations, startingAgent)

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
        totalDistanceFromOptimal = self.getTotalDistanceFromOptimal(agent1items, agent2items, bestprefSum)

        hasParetoOptimal, agent1Score, agent2Score = self.isParetoOptimal(domain)

        f.write(f"Is Pareto Optimal: {hasParetoOptimal}\n")
            
        if hasParetoOptimal:
            print(f"{Fore.GREEN}Is Pareto Optimal{Fore.RESET}\n")
        else:
            print(f"{Fore.RED}Is Not Pareto Optimal{Fore.RESET}\n")
        print(f"Is Pareto Optimal: {hasParetoOptimal}")
        
        if totalDistanceFromOptimal < distanceFromOptimalCeiling:
            f.write(f"\nTotal Distance from Optimal (PASSING): {totalDistanceFromOptimal}% away from the optimal.\n")
            print(f"\n{Fore.GREEN}Total Distance from Optimal (PASSING): {totalDistanceFromOptimal}% away from the optimal.{Fore.RESET}")
        else:
            f.write(f"\nTotal Distance from Optimal (FAILING): {totalDistanceFromOptimal}% away from the optimal.\n")
            print(f"\n{Fore.RED}Total Distance from Optimal (FAILING): {totalDistanceFromOptimal}% away from the optimal.{Fore.RESET}")
            exceededDistFromOptimal = True

        hasOptimalAllocation = self.hasOptimalAllocation(agent1items, agent2items, bestprefSum)
        if not hasOptimalAllocation:
            f.write("\nOptimal Allocation:\n")
            f.write(f"  Agent 1: {optimalAllocation1}\n")
            f.write(f"  Agent 2: {optimalAllocation2}\n")
            
            print("\nOptimal Allocation:\n")
            print(f"    Agent 1: {optimalAllocation1}")
            print(f"    Agent 2: {optimalAllocation2}")
            
        f.write(f"\nNumber of conversation iterations: {domain.numConversationIterations}")
        f.write(f"\nNumber of tokens generated by {domain.agent1.name}: {domain.agent1.numTokensGenerated}")
        f.write(f"\nNumber of tokens generated by {domain.agent2.name}: {domain.agent2.numTokensGenerated}\n")
        f.close()
        
        return items, hasOptimalAllocation, exceededDistFromOptimal, totalDistanceFromOptimal, hasParetoOptimal, agent1Score, agent2Score
    
    def generateRandomPrefValues(self): # Generate random prefValues between 0 and 1
        prefValues = []
        for i in range(self.numItems):  # Generate random tasks
            prefValue = round(random.uniform(0, 1), 1) # Generate random PSRs between 0 and 1, rounded to 1 decimal place
            prefValues.append(prefValue)
        return prefValues
    
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

def add_test_methods(numRounds, numitems, numIterations, distanceFromOptimalCeiling, logFilename, model, useOpenAI):
    numOptimal = 0
    numPassing = 0
    
    agent1ScoreSum = 0 # the percentage of the total preference value score of the items assigned to agent 1
    agent2ScoreSum = 0 # the percentage of the total preference value score of the items assigned to agent 2
    numParetoOptimal = 0
    
    totalAllocationScore = 0
    ta = TestAgent()
    ta.setUp(numitems)
    totalTime = 0

    for i in range(numRounds):
        startTime = time.time()
        items, hasOptimalAllocation, exceededDistFromOptimal, allocationScore, hasParetoOptimal, agent1Score, agent2Score = ta.run_round(i+1, numRounds, numIterations, distanceFromOptimalCeiling, logFilename, model, useOpenAI)
        if hasParetoOptimal:
            numParetoOptimal += 1

        agent1MaxPointsPossible = 0
        agent2MaxPointsPossible = 0
        for item in items:
            agent1MaxPointsPossible += item.pref1
            agent2MaxPointsPossible += item.pref2
        agent1ScoreSum += (agent1Score / agent1MaxPointsPossible) * 100
        agent2ScoreSum += (agent2Score / agent2MaxPointsPossible) * 100
            
        totalAllocationScore += allocationScore
        endTime = time.time()
        duration = endTime - startTime
        totalTime += duration
        numOptimal += 1 if hasOptimalAllocation else 0
        numPassing += 1 if not exceededDistFromOptimal else 0
        with open (logFilename, "a") as f:
            f.write(f"Round {i+1} Duration: Completed in {format_seconds(duration)}\n\n")
            f.close()

    with open(logFilename, "a") as f:
        f.write(("=" * 25) + f"  TOTAL  " + ("=" * 25) + "\n")
        f.write("\nFAIRNESS CRITERIA:\n")
        f.write(f"\nTotal Optimal Allocations: {numOptimal} of {numRounds} ({round(100*(numOptimal / numRounds))}%) rounds were optimal.")
        f.write(f"\nTotal Passing Allocations: {numPassing} of {numRounds} ({round(100*(numPassing / numRounds))}%) rounds were within the allocation test tolerance.")
        f.write(f"\nAverage Total Distance from Optimal: {round(totalAllocationScore/numRounds)}% away from the optimal.")
        f.write("\n")
        
        f.write("\nCOMPETITIVE CRITERIA:\n")
        f.write("\nNumber of Pareto Optimal Allocations: " + str(numParetoOptimal) + " of " + str(numRounds) + " rounds.")
        f.write("\n")
        f.write(f"\nAverage Agent 1 Allocation Score: {(agent1ScoreSum / numRounds)}%")
        f.write(f"\nAverage Agent 2 Allocation Score: {(agent2ScoreSum / numRounds)}%")
        f.write("\n     Note: The Allocation Score for each round is the sum of an agent's preference values out of the sum of all possible preference values for that agent.")
        f.write("\n")
        
        f.write(f"\nTotal Time (hh:mm:ss): {format_seconds(totalTime)}")
        f.write(f"\nAverage Time per Round (hh:mm:ss): {format_seconds(totalTime/numRounds)}\n")
        f.close()

main()