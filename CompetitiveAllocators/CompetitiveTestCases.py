import unittest
import random
from colorama import Fore
import CompetitiveTaskAllocator as compTA
import time
import itertools
from datetime import datetime

def main():
    numRounds = 100 # Number of rounds to be run
    numitems = 4 # Number of items to be assigned per round
    numIterations = 4 # Number of conversation iterations per round
    distanceFromOptimalCeiling = 15 # The maximum percentage away from the optimal allocation that is considered passing
    model = 'llama3.1:70b' # The LLM model to be used for allocation
    
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
    add_test_methods(numRounds, numitems, numIterations, distanceFromOptimalCeiling, logFilename, model)

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
    
    def getOneSidedAllocationScore(self, domain, agentName, agentItems, items): # calculates allocation score by summing up the preference values of the items assigned to the agent 
        runningScore = 0
        totalPossible = 0
        if agentName == domain.agent1.name:
            for item in items:
                totalPossible += item.pref1
            for item in agentItems:
                runningScore += item.pref1
        elif agentName == domain.agent2.name:
            for item in items:
                totalPossible += item.pref2
            for item in agentItems:
                runningScore += item.pref2
        return round(100 * (runningScore / totalPossible))
        

    def setUp(self, numItems):
        self.numItems = numItems  # number of items to be assigned

    def run_round(self, round_num, numRounds, numIterations, distanceFromOptimalCeiling, logFilename, model):
        exceededDistFromOptimal = False

        print("\n" + ("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")
        with open(logFilename, "a") as f:
            f.write("\n" + ("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")
            f.close()

        # Generate items and prefSums
        items = [] # list of Item objects
        itemNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
        
        # Asymmetric Preference Values - easiest case
        prefValues1 = [0.1, 0.3, 0.6, 1.0]
        prefValues2 = [1.0, 0.6, 0.3, 0.1]
        
        # Random Preference Values
        # prefValues1 = self.generateRandomPrefValues()
        # prefValues2 = self.generateRandomPrefValues()
        
        for i in range(self.numItems):  # Generate items
            item = f"Item {itemNames[i]}"
            pref1 = prefValues1[i]
            pref2 = prefValues2[i]
            items.append(compTA.Item(item, pref1, pref2))
            
        domain = compTA.Domain(items, model)

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
        agent1AllocationScore = self.getOneSidedAllocationScore(domain, domain.agent1.name, agent1items, items)
        agent2AllocationScore = self.getOneSidedAllocationScore(domain, domain.agent2.name, agent2items, items)
        
        if totalDistanceFromOptimal < distanceFromOptimalCeiling:
            f.write(f"\nTotal Distance from Optimal (PASSING): {totalDistanceFromOptimal}% away from the optimal.\n")
            print(f"\n{Fore.GREEN}Total Distance from Optimal (PASSING): {totalDistanceFromOptimal}% away from the optimal.{Fore.RESET}")
        else:
            f.write(f"\nTotal Distance from Optimal (FAILING): {totalDistanceFromOptimal}% away from the optimal.\n")
            print(f"\n{Fore.RED}Total Distance from Optimal (FAILING): {totalDistanceFromOptimal}% away from the optimal.{Fore.RESET}")
            exceededDistFromOptimal = True
        
        if agent1AllocationScore > agent2AllocationScore:
            winner = domain.agent1.name
            f.write(f"\n{domain.agent1.name} wins with a {agent1AllocationScore}% allocation score.")
            f.write(f"\n{domain.agent2.name} has a {agent2AllocationScore}% allocation score.")
            print(f"\n{domain.agent1.name} wins with a {agent1AllocationScore}% allocation score.")
            print(f"{domain.agent2.name} has a {agent2AllocationScore}% allocation score.")
        elif agent2AllocationScore > agent1AllocationScore:
            winner = domain.agent2.name
            f.write(f"\n{domain.agent2.name} wins with a {agent2AllocationScore}% allocation score.")
            f.write(f"\n{domain.agent1.name} has a {agent1AllocationScore}% allocation score.")
            print(f"\n{domain.agent2.name} wins with a {agent2AllocationScore}% allocation score.")
            print(f"{domain.agent1.name} has a {agent1AllocationScore}% allocation score.")
        else:
            winner = "Tie"
            f.write(f"\nIt's a tie! Both agents have an allocation score of {agent1AllocationScore}%.")
            print(f"\nIt's a tie! Both agents have an allocation score of {agent1AllocationScore}%.")
        
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
        
        return hasOptimalAllocation, exceededDistFromOptimal, totalDistanceFromOptimal, winner, domain.agent1.name, agent1AllocationScore, domain.agent2.name, agent2AllocationScore
    
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

def add_test_methods(numRounds, numitems, numIterations, distanceFromOptimalCeiling, logFilename, model):
    numOptimal = 0
    numPassing = 0
    
    agent1Wins = 0
    agent2Wins = 0
    agent1ScoreSum = 0
    agent2ScoreSum = 0
    ties = 0
    
    totalAllocationScore = 0
    ta = TestAgent()
    ta.setUp(numitems)
    totalTime = 0

    for i in range(numRounds):
        startTime = time.time()
        hasOptimalAllocation, exceededDistFromOptimal, allocationScore, winner, agent1Name, agent1Score, agent2Name, agent2Score = ta.run_round(i+1, numRounds, numIterations, distanceFromOptimalCeiling, logFilename, model)
        if winner == agent1Name:
            agent1Wins += 1
            agent1ScoreSum += agent1Score
        elif winner == agent2Name:
            agent2Wins += 1
            agent2ScoreSum += agent2Score
        elif winner == "Tie":
            ties += 1
            agent1ScoreSum += agent1Score
            agent2ScoreSum += agent2Score
        else:
            raise ValueError("Winner not found.")
            
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
        f.write(f"\nTotal Optimal Allocations: {numOptimal} of {numRounds} ({round(100*(numOptimal / numRounds))}%) rounds were optimal.")
        f.write(f"\nTotal Passing Allocations: {numPassing} of {numRounds} ({round(100*(numPassing / numRounds))}%) rounds were within the allocation test tolerance.")
        f.write(f"\nAverage Total Distance from Optimal: {round(totalAllocationScore/numRounds)}% away from the optimal.")
        f.write("\n")
        
        f.write(f"\nAgent 1 Wins: {agent1Name} won {agent1Wins} of {numRounds} ({100*(round(agent1Wins / numRounds))}%) rounds")
        f.write(f"\nAgent 2 Wins: {agent2Name} won {agent2Wins} of {numRounds} ({100*(round(agent2Wins / numRounds))}%) rounds")
        f.write(f"\nTies: Agents tied in {ties} of {numRounds} ({round(100*(ties / numRounds))}%) rounds")
        f.write(f"\nAverage Agent 1 Allocation Score: {round(agent1ScoreSum / numRounds)}%")
        f.write(f"\nAverage Agent 2 Allocation Score: {round(agent2ScoreSum / numRounds)}%")
        f.write("\n")
        
        f.write(f"\nTotal Time (hh:mm:ss): {format_seconds(totalTime)}")
        f.write(f"\nAverage Time per Round (hh:mm:ss): {format_seconds(totalTime/numRounds)}\n")
        f.close()

main()