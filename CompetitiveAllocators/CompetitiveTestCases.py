import random
import time
import itertools
from datetime import datetime
from colorama import Fore
import CompetitiveTaskAllocator as compTA
import re  # Add this import for regular expression functions

def main():
    numRounds = 1  # Number of rounds to be run
    numItems = 6  # Number of items to be assigned per round
    maxIterations = 32  # Number of conversation iterations per round
    distanceFromOptimalCeiling = 15  # Example value

    agent1Model = "gemma2:latest"  # Model for agent 1
    agent1UseOpenAI = False  # Use OpenAI API for agent 1
    # Let agent 1 be a "default" agent with no strategy.

    agent2Model = "gemma2:latest"  # Model for agent 2
    agent2UseOpenAI = False  # Use OpenAI API for agent 2
    # Let agent 2 be a modified agent using the Boulware strategy.

    moderatorModel = "gemma2:latest"  # Model for the moderator
    moderatorUseOpenAI = False  # Use OpenAI API for the moderator

    print("\n" + ("=" * 25) + "  COMPETITIVE ITEM ALLOCATION TEST  " + ("=" * 25) + "\n")
    print(f"Number of Items to Allocate: {numItems}")
    print(f"Number of Rounds: {numRounds}")
    print(f"Maximum Number of Conversation Iterations Per Round: {maxIterations}")
    print(f"Total Distance from Optimal Ceiling: Allocations must be less than {distanceFromOptimalCeiling}% away from the optimal solution")
    print(f"\nAgent 1 LLM Being Used to Allocate items: {agent1Model}")
    print(f"Agent 2 LLM Being Used to Allocate items: {agent2Model}\n")

    # Sanitize model names for filenames
    safeAgent1Model = re.sub(r'[<>:"/\\|?*]', '_', agent1Model)
    safeAgent2Model = re.sub(r'[<>:"/\\|?*]', '_', agent2Model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logFilename = f"{safeAgent1Model}_{safeAgent2Model}_{numItems}Items_{timestamp}.txt"

    with open(logFilename, "w") as f:
        f.write("COMPETITIVE ITEM ALLOCATION LOG\n\n")
        f.write(f"Number of Items to Allocate: {numItems}\n")
        f.write(f"Number of Rounds: {numRounds}\n")
        f.write(f"Maximum Number of Conversation Iterations Per Round: {maxIterations}\n")
        f.write(f"Total Distance from Optimal Ceiling: Allocations must be less than {distanceFromOptimalCeiling}% away from the optimal solution\n")
        f.write(f"Agent 1 LLM Being Used to Allocate items: {agent1Model}")
        f.write(f"Agent 2 LLM Being Used to Allocate items: {agent2Model}\n")

    run_tests(numRounds, numItems, maxIterations, distanceFromOptimalCeiling, logFilename, agent1Model, agent2Model, moderatorModel, agent1UseOpenAI, agent2UseOpenAI, moderatorUseOpenAI)

def run_tests(numRounds, numItems, maxIterations, distanceFromOptimalCeiling, logFilename, agent1Model, agent2Model, moderatorModel, agent1UseOpenAI, agent2UseOpenAI, moderatorUseOpenAI):
    numOptimal = 0
    numPassing = 0
    numParetoOptimal = 0
    totalAllocationScore = 0
    agent1ScoreSum = 0
    agent2ScoreSum = 0
    totalTime = 0

    for round_num in range(1, numRounds + 1):
        startTime = time.time()

        print("\n" + ("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")
        with open(logFilename, "a") as f:
            f.write("\n" + ("~" * 25) + f"  ROUND {round_num} OF {numRounds}  " + ("~" * 25) + "\n")

        items = generate_items(numItems)
        domain = compTA.Domain(items, agent1Model, agent2Model, moderatorModel, agent1UseOpenAI, agent2UseOpenAI, moderatorUseOpenAI)
        print("\nItems for this round:")
        print(items)
        startingAgent = domain.agent1 if round_num % 2 == 1 else domain.agent2
        domain.startNegotiation(maxIterations, startingAgent)

        agent1items = domain.boardState.getItems(domain.agent1.name)
        agent2items = domain.boardState.getItems(domain.agent2.name)

        domain.printItems()

        with open(logFilename, "a") as f:
            f.write("\nItems for this round:\n")
            for item in items:
                f.write(f"- {item}\n")
            compTA.logAssignedItems(logFilename, domain.boardState)

        optimalAllocation1, optimalAllocation2, bestPrefSum = getOptimalAllocation(items)
        totalDistanceFromOptimal = getTotalDistanceFromOptimal(agent1items, agent2items, bestPrefSum)
        hasParetoOptimal, agent1Score, agent2Score = isParetoOptimal(domain)

        with open(logFilename, "a") as f:
            f.write(f"Is Pareto Optimal: {hasParetoOptimal}\n")

        if hasParetoOptimal:
            print(f"{Fore.GREEN}Is Pareto Optimal{Fore.RESET}\n")
        else:
            print(f"{Fore.RED}Is Not Pareto Optimal{Fore.RESET}\n")
        print(f"Is Pareto Optimal: {hasParetoOptimal}")

        # Display the utility graph after Pareto optimality check
        domain.generate_utility_plot()

        if totalDistanceFromOptimal < distanceFromOptimalCeiling:
            with open(logFilename, "a") as f:
                f.write(f"\nTotal Distance from Optimal (PASSING): {totalDistanceFromOptimal}% away from the optimal.\n")
            print(f"\n{Fore.GREEN}Total Distance from Optimal (PASSING): {totalDistanceFromOptimal}% away from the optimal.{Fore.RESET}")
        else:
            with open(logFilename, "a") as f:
                f.write(f"\nTotal Distance from Optimal (FAILING): {totalDistanceFromOptimal}% away from the optimal.\n")
            print(f"\n{Fore.RED}Total Distance from Optimal (FAILING): {totalDistanceFromOptimal}% away from the optimal.{Fore.RESET}")

        hasOptimal = hasOptimalAllocation(agent1items, agent2items, bestPrefSum)
        if not hasOptimal:
            with open(logFilename, "a") as f:
                f.write("\nOptimal Allocation:\n")
                f.write(f"  Agent 1: {optimalAllocation1}\n")
                f.write(f"  Agent 2: {optimalAllocation2}\n")

            print("\nOptimal Allocation:\n")
            print(f"    Agent 1: {optimalAllocation1}")
            print(f"    Agent 2: {optimalAllocation2}")

        with open(logFilename, "a") as f:
            f.write(f"\nNumber of conversation iterations: {domain.numConversationIterations}")
            f.write(f"\nNumber of tokens generated by {domain.agent1.name}: {domain.agent1.numTokensGenerated}")
            f.write(f"\nNumber of tokens generated by {domain.agent2.name}: {domain.agent2.numTokensGenerated}\n")

        # Removed undefined currentAgent and currentInput usage

        endTime = time.time()
        duration = endTime - startTime  # Define duration
        totalTime += duration
        numOptimal += 1 if hasOptimal else 0
        numPassing += 1 if totalDistanceFromOptimal < distanceFromOptimalCeiling else 0
        numParetoOptimal += 1 if hasParetoOptimal else 0

        agent1MaxPointsPossible = sum(item.pref1 for item in items)
        agent2MaxPointsPossible = sum(item.pref2 for item in items)
        agent1ScoreSum += (agent1Score / agent1MaxPointsPossible) * 100
        agent2ScoreSum += (agent2Score / agent2MaxPointsPossible) * 100
        totalAllocationScore += totalDistanceFromOptimal

        with open(logFilename, "a") as f:
            f.write(f"Round {round_num} Duration: Completed in {format_seconds(duration)}\n\n")

    with open(logFilename, "a") as f:
        f.write(("=" * 25) + "  TOTAL  " + ("=" * 25) + "\n")
        f.write("\nFAIRNESS CRITERIA:\n")
        f.write(f"\nTotal Optimal Allocations: {numOptimal} of {numRounds} ({round(100*(numOptimal / numRounds))}%) rounds were optimal.")
        f.write(f"\nTotal Passing Allocations: {numPassing} of {numRounds} ({round(100*(numPassing / numRounds))}%) rounds were within the allocation test tolerance.")
        f.write(f"\nAverage Total Distance from Optimal: {round(totalAllocationScore / numRounds)}% away from the optimal.\n")
        f.write("\nCOMPETITIVE CRITERIA:\n")
        f.write(f"\nNumber of Pareto Optimal Allocations: {numParetoOptimal} of {numRounds} rounds.")
        f.write(f"\nAverage Agent 1 Allocation Score: {agent1ScoreSum / numRounds}%")
        f.write(f"\nAverage Agent 2 Allocation Score: {agent2ScoreSum / numRounds}%")
        f.write("\n     Note: The Allocation Score for each round is the sum of an agent's preference values out of the sum of all possible preference values for that agent.\n")
        f.write(f"\nTotal Time (hh:mm:ss): {format_seconds(totalTime)}")
        f.write(f"\nAverage Time per Round (hh:mm:ss): {format_seconds(totalTime / numRounds)}\n")

def generate_items(numItems):
    itemNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    prefValues1 = [round(random.uniform(0, 1), 1) for _ in range(numItems)]
    prefValues2 = [round(random.uniform(0, 1), 1) for _ in range(numItems)]
    items = []
    for i in range(numItems):
        item = f"Item {itemNames[i]}"
        items.append(compTA.Item(item, prefValues1[i], prefValues2[i]))
    return items


def getOptimalAllocation(items):
    optimalSolution = []
    bestPrefSum = 0
    n = len(items)

    if n % 2 != 0:
        raise ValueError("The number of items must be even to form two equal groups.")

    all_combinations = [comb for comb in itertools.combinations(items, n // 2)]

    for comb in all_combinations:
        group1 = comb
        group2 = tuple(item for item in items if item not in group1)

        prefSum = calculatePrefSum(group1, group2)
        if prefSum > bestPrefSum:
            bestPrefSum = prefSum
            optimalSolution = [group1, group2]

    return optimalSolution[0], optimalSolution[1], bestPrefSum
        
def calculatePrefSum(agent1items, agent2items):
    p1 = sum(item.pref1 for item in agent1items)
    p2 = sum(item.pref2 for item in agent2items)
    return p1 + p2

def hasOptimalAllocation(agent1items, agent2items, bestPrefSum):
    currPrefSum = calculatePrefSum(agent1items, agent2items)
    return currPrefSum >= bestPrefSum

def getTotalDistanceFromOptimal(agent1items, agent2items, bestPrefSum):
    currPrefSum = calculatePrefSum(agent1items, agent2items)
    return round(100 * abs(currPrefSum - bestPrefSum) / bestPrefSum)

def isParetoOptimal(domain):
    agent1items = domain.boardState.getItems(domain.agent1.name)
    agent2items = domain.boardState.getItems(domain.agent2.name)
    items = domain.items
    
    # Calculate current utilities
    currentScore1 = sum(item.pref1 for item in agent1items)
    currentScore2 = sum(item.pref2 for item in agent2items)
    
    # Generate all possible allocations
    n = len(items)
    epsilon = 1e-6  # Small value for floating-point comparison
    
    def is_nearly_greater(a, b, epsilon=epsilon):
        return a > b + epsilon

    for i in range(n + 1):  # Try all possible group sizes
        for alternative_agent1_items in itertools.combinations(items, i):
            # Create the complementary allocation for agent 2
            alternative_agent2_items = tuple(item for item in items if item not in alternative_agent1_items)
            
            # Calculate utilities for this alternative allocation
            altScore1 = sum(item.pref1 for item in alternative_agent1_items)
            altScore2 = sum(item.pref2 for item in alternative_agent2_items)
            
            # Check if this alternative allocation dominates the current one
            if ((is_nearly_greater(altScore1, currentScore1) and (altScore2 > currentScore2 - epsilon)) or
                (is_nearly_greater(altScore2, currentScore2) and (altScore1 > currentScore1 - epsilon))):
                
                print(f"\nNot Pareto Optimal.\n")
                print(f"{Fore.YELLOW}Current Allocation:")
                print(f"Agent 1: {agent1items} (utility: {currentScore1})")
                print(f"Agent 2: {agent2items} (utility: {currentScore2}){Fore.RESET}")
                print(f"\n{Fore.RED}Dominating Allocation:")
                print(f"Agent 1: {alternative_agent1_items} (utility: {altScore1})")
                print(f"Agent 2: {alternative_agent2_items} (utility: {altScore2}){Fore.RESET}")
                return False, currentScore1, currentScore2
    
    return True, currentScore1, currentScore2

def format_seconds(seconds):
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{remaining_seconds:02}"
if __name__ == '__main__':    main()

# Need to slow down how quickly boulware agent decreases index. Make it fit expected curve
# Fix bug where boulware doesnt behave correctly.
# Agent 2 doesn't consistently listen to system messages telling it what to offer
# Agent 2 says deal when system message tells them to make new offer
# add check for new line character after "PROPOSAL:" in parse_allocation