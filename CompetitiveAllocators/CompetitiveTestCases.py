import unittest
import random
from colorama import Fore
import CompetitiveTaskAllocator as compTA
import time
import itertools
from datetime import datetime

def main():
    numRounds = 1 # Number of rounds to be run
    numitems = 4 # Number of items to be assigned per round
    numIterations = 4 # Number of conversation iterations per round
    distanceFromOptimalCeiling = 15 # The maximum percentage away from the optimal allocation that is considered passing
    model = 'gemma2:latest' # The LLM model to be used for allocation
    
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
        
    def getParetoOptimalRanking(self, domain, agentName, items):
        all_combinations = []
        all_allocations = []
        agentItems = []
        
        if agentName == domain.agent1.name:
            agentItems = domain.boardState.getItems(domain.agent1.name)
        elif agentName == domain.agent2.name:
            agentItems = domain.boardState.getItems(domain.agent2.name)
        else:
            raise ValueError("Agent not found.")
        
        for r in range(0, len(items)):
            all_combinations.extend(itertools.combinations(items, r))
            
            
        for comb in all_combinations:
            group1 = comb
            group2 = tuple(item for item in items if item not in group1)
            
            # calculate agent X's score for this allocation and add it to all_allocations
            agentScore = 0
            if agentName == domain.agent1.name:
                for item in group1:
                    agentScore += item.pref1
            elif agentName == domain.agent2.name:
                for item in group2:
                    agentScore += item.pref2
            else:
                raise ValueError("Agent not found.")

            all_allocations.append((group1, group2, agentScore))
        
        # code to sort allocations of items in descending (highest to lowest) order of agent's preference values
        sorted_Allocations = sorted(all_allocations, key=lambda allocation: allocation[2], reverse=True) # formatted as [(group1, group2, agentScore), ...]
        agentRank = 0
        for i, allocation in enumerate(sorted_Allocations):
            hasAllItems = True
            
            for item in agentItems:
                if agentName == domain.agent1.name:
                    if len(allocation[0]) != len(agentItems):
                        hasAllItems = False
                    if item not in allocation[0]:
                        hasAllItems = False
                elif agentName == domain.agent2.name:
                    if len(allocation[1]) != len(agentItems):
                        hasAllItems = False
                    if item not in allocation[1]:
                        hasAllItems = False
                
            if hasAllItems:
                agentRank = i + 1
                break
        if agentRank == 0:
            raise ValueError("Agent's ranking not found.")
        
        current_allocation = sorted_Allocations[agentRank - 1]
        current_score_X = current_allocation[2] # current agent X's score
        current_score_Y = sum(item.pref2 for item in current_allocation[1]) if agentName == domain.agent1.name else sum(item.pref1 for item in current_allocation[0]) # current agent Y's score
        
        if agentRank > 1:
            #	For the allocations that are better than the current allocation, see if any of them can increase agent X’s score without decreasing agent Y’s score. 
            #	    If one exists, then you do not have the Pareto optimal. 
            #	    If no such allocation exists, you’ve found the pareto optimal.
            #	Report the ranking of the allocation as well as Agent X’s score.

            for allocation in sorted_Allocations[:agentRank - 1]: # check all allocations better than the current one
                score_X = allocation[2]
                score_Y = sum(item.pref2 for item in allocation[1]) if agentName == domain.agent1.name else sum(item.pref1 for item in allocation[0])

                if score_X > current_score_X and score_Y >= current_score_Y:
                    return agentRank, current_score_X, False  # Not Pareto optimal
            
        
        return agentRank, current_score_X, True  # Pareto optimal
        
    
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
        agent1Rank, agent1Score, hasParetoOptimal1 = self.getParetoOptimalRanking(domain, domain.agent1.name, items)
        agent2Rank, agent2Score, hasParetoOptimal2 = self.getParetoOptimalRanking(domain, domain.agent2.name, items)
    

        totalDistanceFromOptimal = self.getTotalDistanceFromOptimal(agent1items, agent2items, bestprefSum)
        # agent1AllocationScore = self.getOneSidedAllocationScore(domain, domain.agent1.name, agent1items, items)
        # agent2AllocationScore = self.getOneSidedAllocationScore(domain, domain.agent2.name, agent2items, items)
        
        f.write(f"\nAgent 1 Pareto Optimal Ranking: {agent1Rank} with a total preference value score of {agent1Score:.2f}. \n       Is Pareto Optimal: {hasParetoOptimal1}\n")
        f.write(f"\nAgent 2 Pareto Optimal Ranking: {agent2Rank} with a total preference value score of {agent2Score:.2f}. \n       Is Pareto Optimal: {hasParetoOptimal2}\n")
        print(f"\nAgent 1 Pareto Optimal Ranking: {agent1Rank} with a total preference value score of {agent1Score:.2f}. \n         Is Pareto Optimal: {hasParetoOptimal1}")
        print(f"Agent 2 Pareto Optimal Ranking: {agent2Rank} with a total preference value score of {agent2Score:.2f}. \n       Is Pareto Optimal: {hasParetoOptimal2}")
        
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
        
        return items, hasOptimalAllocation, exceededDistFromOptimal, totalDistanceFromOptimal, agent1Rank, agent1Score, hasParetoOptimal1, agent2Rank, agent2Score, hasParetoOptimal2
    
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
    
    agent1RankSum = 0 # the rank of the items assigned to agent 1
    agent1ScoreSum = 0 # the percentage of the total preference value score of the items assigned to agent 1
    agent2RankSum = 0 # the rank of the items assigned to agent 2
    agent2ScoreSum = 0 # the percentage of the total preference value score of the items assigned to agent 2
    numParetoOptimalAgent1 = 0
    numParetoOptimalAgent2 = 0
    
    totalAllocationScore = 0
    ta = TestAgent()
    ta.setUp(numitems)
    totalTime = 0

    for i in range(numRounds):
        startTime = time.time()
        items, hasOptimalAllocation, exceededDistFromOptimal, allocationScore, agent1Rank, agent1Score, hasParetoOptimal1, agent2Rank, agent2Score, hasParetoOptimal2 = ta.run_round(i+1, numRounds, numIterations, distanceFromOptimalCeiling, logFilename, model)
        if hasParetoOptimal1:
            numParetoOptimalAgent1 += 1
        if hasParetoOptimal2:
            numParetoOptimalAgent2 += 1
            
        agent1MaxPointsPossible = 0
        agent2MaxPointsPossible = 0
        for item in items:
            agent1MaxPointsPossible += item.pref1
            agent2MaxPointsPossible += item.pref2
        agent1ScoreSum += (agent1Score / agent1MaxPointsPossible) * 100
        agent1RankSum += agent1Rank
        agent2ScoreSum += (agent2Score / agent2MaxPointsPossible) * 100
        agent2RankSum += agent2Rank
            
            
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
        f.write(f"\nAverage Agent 1 Pareto Optimal Ranking: {(agent1RankSum / numRounds)}")
        f.write(f"\nAverage Agent 2 Pareto Optimal Ranking: {(agent2RankSum / numRounds)}")
        f.write("\n")
        f.write("\nAgent 1 Pareto Optimal Allocations: " + str(numParetoOptimalAgent1) + " of " + str(numRounds) + " rounds.")
        f.write("\nAgent 2 Pareto Optimal Allocations: " + str(numParetoOptimalAgent2) + " of " + str(numRounds) + " rounds.")
        f.write("\n")
        f.write(f"\nAverage Agent 1 Allocation Score: {(agent1ScoreSum / numRounds)}%")
        f.write(f"\nAverage Agent 2 Allocation Score: {(agent2ScoreSum / numRounds)}%")
        f.write("\nNote: The Allocation Score is, for each round, the sum of an agent's preference values out of the sum of all possible preference values for that agent.")
        f.write("\n")
        
        f.write(f"\nTotal Time (hh:mm:ss): {format_seconds(totalTime)}")
        f.write(f"\nAverage Time per Round (hh:mm:ss): {format_seconds(totalTime/numRounds)}\n")
        f.close()

main()