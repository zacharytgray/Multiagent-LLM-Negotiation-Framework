from agent import Agent
from task import Task
from colorama import Fore
from proposal import Proposal
from negotiationFlag import NegotiationFlag
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import random
import datetime
import copy
import ast

class Negotiation:
    def __init__(self, roundIndex, numTasks, maxIterations, agent1Model, agent1usesOpenAI, agent1Type, agent2Model, agent2usesOpenAI, agent2Type, agent1Name, agent2Name, hasInitialProposal):
        self.roundIndex = roundIndex
        self.DNF = False # Did Not Finish
        self.numTasks = numTasks
        self.agent1 = Agent(agentName=agent1Name, modelName=agent1Model, usesOpenAI=agent1usesOpenAI, agentType=agent1Type)
        self.agent2 = Agent(agentName=agent2Name, modelName=agent2Model, usesOpenAI=agent2usesOpenAI, agentType=agent2Type)
        self.numIterations = 0 # Number of conversation iterations in the negotiation
        self.maxIterations = maxIterations
        self.hasInitialProposal = hasInitialProposal
        self.initialProposal = None # The initial proposal, set in setUpInitialProposal()
        self.tasks = self.generateTasks(self.numTasks) # or use generateRandomTasks() for random tasks
        # self.tasks = self.generateRandomTasks(self.numTasks)
        self.negotiationTime = 0 # Time taken for the negotiation
        self.winningProposal = None # The winning proposal at the end of the negotiation
        
    def updateAgentInstructions(self, currentAgent): # Add the negotiation tasks to the agent's instructions
        
        self.agent1.systemInstructions += "\n**NOW, HERE ARE THE ACTUAL ITEMS YOU MUST ALLOCATE:**\n"
        self.agent2.systemInstructions += "\n**NOW, HERE ARE THE ACTUAL ITEMS YOU MUST ALLOCATE:**\n"
        for task in self.tasks:
            self.agent1.systemInstructions += f"{task.mappedName}: Your confidence level for this task is {task.confidence1}.\n"
            self.agent2.systemInstructions += f"{task.mappedName}: Your confidence level for this task is {task.confidence2}.\n"
        
        if self.hasInitialProposal:
            combinedTaskStr = self.setUpInitialProposal()
            currentAgent.systemInstructions += f"""
\n\n**VERY IMPORTANT!** 
FOR THIS ROUND ONLY, You should start by formally proposing the following initial allocation using the JSON format with the 'formal_proposal' keyword:
formal_proposal = {{
    '{self.agent1.agentName}_items': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent1Tasks])}],
    '{self.agent2.agentName}_items': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent2Tasks])}],
    'has_deal': 'False'
}}\n
COPY THIS ALLOCATION EXACTLY, ESPECIALLY TASK ASSIGNMENTS, AND DO NOT CHANGE ANYTHING ELSE.
"""
        self.agent1.systemInstructions = self.agent1.systemInstructions.replace("[my_name]", self.agent1.agentName).replace("[opponent_name]", self.agent2.agentName)
        self.agent2.systemInstructions = self.agent2.systemInstructions.replace("[my_name]", self.agent2.agentName).replace("[opponent_name]", self.agent1.agentName)
        
    def setUpInitialProposal(self):
        shuffledTasks = copy.deepcopy(self.tasks)
        random.shuffle(shuffledTasks)
        agent1Tasks = shuffledTasks[:self.numTasks//2]
        agent2Tasks = shuffledTasks[self.numTasks//2:self.numTasks]
        self.initialProposal = Proposal(agent1Tasks, agent2Tasks)
        combinedTaskStr = f"\n{self.agent1.agentName}: {', '.join([task.mappedName for task in agent1Tasks])}\n{self.agent2.agentName}: {', '.join([task.mappedName for task in agent2Tasks])}"
        return combinedTaskStr
        
    def generateTasks(self, numTasks):
        baseTasks = [
            Task(name="Task A", pref1=0.5, pref2=0.7),
            Task(name="Task B", pref1=0.6, pref2=0.2),
            Task(name="Task C", pref1=0.1, pref2=0.9),
            Task(name="Task D", pref1=0.8, pref2=0.4),
            Task(name="Task E", pref1=0.5, pref2=0.3),
            Task(name="Task F", pref1=0.2, pref2=0.6),
            Task(name="Task G", pref1=0.9, pref2=0.1),
            Task(name="Task H", pref1=0.4, pref2=0.8),
            Task(name="Task I", pref1=0.7, pref2=0.5),
            Task(name="Task J", pref1=0.0, pref2=1.0),
            Task(name="Task K", pref1=1.0, pref2=0.0),
            Task(name="Task L", pref1=0.3, pref2=0.5), #12 tasks
        ]
        return baseTasks[:numTasks]
    
    def generateRandomTasks(self, numTasks):
        tasks = []
        for i in range(numTasks):
            taskName = chr(65 + i)  # Generate task names "Task A", "Task B", etc.
            pref1 = round(random.uniform(0.0, 1.0), 1)
            pref2 = round(random.uniform(0.0, 1.0), 1)
            task = Task(name=f"Task {taskName}", pref1=pref1, pref2=pref2)
            tasks.append(task)
        return tasks
    
    def startNegotiation(self):
        negotiationStartTime = datetime.datetime.now().replace(microsecond=0)
        print(f"\n{Fore.GREEN}Round {self.roundIndex} started{Fore.RESET}")
        startingAgent = random.choice([self.agent1, self.agent2]) # Randomly select the starting agent

        currentAgent, otherAgent = (self.agent1, self.agent2) if startingAgent == self.agent1 else (self.agent2, self.agent1)
            
        self.updateAgentInstructions(currentAgent) # Add the negotiation tasks to the agent's instructions
        self.agent1.addToChatHistory('system', self.agent1.systemInstructions)
        self.agent2.addToChatHistory('system', self.agent2.systemInstructions)
        print(f"initial proposal: {self.initialProposal}")
            
        currentInput = f"Hello, I am {otherAgent.agentName}. Let's begin the task negotiation. Please start the negotiation process."
        agreementReached = False
        dealCounter = 0
        
        # print tasks for this negotiation
        print("Tasks for this negotiation:")
        for task in self.tasks:
            print(f"{task.mappedName}: {task.confidence1}, {task.confidence2}")
        
        while not agreementReached and self.numIterations < self.maxIterations:
            maxRetries = 5
            retries = 0
            while retries < maxRetries: # Keep retrying if the proposal is invalid
                # if retries > 0:
                #     currentAgent.printMemory()
                currentResponse = currentAgent.generateResponse('user', currentInput)
                potentialProposal = self.extractProposalFromReponse(currentResponse)
                if potentialProposal != NegotiationFlag.PROPOSAL_NOT_FOUND: # If proposal is found, check if it is valid
                    if (self.hasInitialProposal and self.numIterations == 0) \
                        and not self.doesProposalMatchInitialProposal(potentialProposal): 
                            # if we want an initial allocation and it's the first round and proposal does not match initial proposal,
                            # reprompt agent for proposal, matching the initial proposal this time.
                            print(f"{Fore.RED}Invalid proposal: Does Not Match Initial Proposal{Fore.RESET}")
                            helperMessage = f"IMPORTANT: YOU ENTERED AN INVALID PROPOSAL. Just this once, you must propose the given initial proposal with the 'PROPSAL:' keyword. Remember, the initial proposal is:\n{self.agent1.agentName}: {', '.join([task.mappedName for task in self.initialProposal.agent1Tasks])}\n{self.agent2.agentName}: {', '.join([task.mappedName for task in self.initialProposal.agent2Tasks])}\nTry Again"
                            currentAgent.addToChatHistory('system', helperMessage)  
                    else:  
                        isValidProposal = potentialProposal.validateProposal(self.tasks)
                        if isValidProposal == NegotiationFlag.ERROR_FREE: # If proposal is valid, update the current agent's proposal
                            currentAgent.proposal = potentialProposal # Update currentAgent.proposal
                            break # Exit the retry loop when a valid proposal is found
                        else: # If proposal is invalid, prompt the agent to correct it
                            # Remove the last incorrect response from the memory
                            # currentAgent.memory = currentAgent.memory[:-1]
                            if isValidProposal == NegotiationFlag.NOT_ENOUGH_TASKS:
                                # reprompt agent for proposal, including all tasks this time.
                                print(f"{Fore.RED}Invalid proposal: Not Enough Tasks{Fore.RESET}")
                                helperMessage = f"IMPORTANT: You must include all {len(self.tasks)} tasks in your proposal. Remember, these are the tasks for this negotiation: {', '.join([task.mappedName for task in self.tasks])}\nTry Again"
                                currentAgent.addToChatHistory('system', helperMessage)
                            elif isValidProposal == NegotiationFlag.TOO_MANY_TASKS:
                                # reprompt agent for proposal, including only given tasks this time.
                                print(f"{Fore.RED}Invalid proposal: Too Many Tasks{Fore.RESET}")
                                helperMessage = f"IMPORTANT: You must include only {len(self.tasks)} tasks in your proposal. Remember, these are the tasks for this negotiation: {', '.join([task.mappedName for task in self.tasks])}.\nTry Again"
                                currentAgent.addToChatHistory('system', helperMessage)
                            elif isValidProposal == NegotiationFlag.INVALID_TASKS_PRESENT:
                                # reprompt agent for proposal, including only given tasks this time.
                                print(f"{Fore.RED}Invalid proposal: Invalid Tasks Present{Fore.RESET}")
                                helperMessage = f"IMPORTANT: You must include only the following {len(self.tasks)} tasks in your proposal: {', '.join([task.mappedName for task in self.tasks])}.\nTry Again"
                                currentAgent.addToChatHistory('system', helperMessage)      
                            else:
                                print(f"{Fore.RED}Invalid proposal: Unknown Error{Fore.RESET}")   
                                print(f"{Fore.RED}Proposal that caused error: \n{potentialProposal}{Fore.RESET}")   
                    retries += 1
                    print(f"{Fore.RED}Retrying... ({retries}/{maxRetries} retries){Fore.RESET}")
                elif potentialProposal == NegotiationFlag.INVALID_PROPOSAL_FORMAT:
                    print(f"{Fore.RED}Invalid proposal: Invalid Proposal Format{Fore.RESET}")
                    helperMessage = f"""IMPORTANT: You must use the following JSON format with NO exceptions:
                    formal_proposal = {{
                        '{currentAgent.agentName}_items': ['Task A', 'Task B'],
                        '{otherAgent.agentName}_items': ['Task C', 'Task D'],
                        'has_deal': 'True'
                    }}
                    Replace the alphabetized task names with the actual task names you want to propose.
                    \nTry Again"""
                    currentAgent.addToChatHistory('system', helperMessage)
                    retries += 1
                    print(f"{Fore.RED}Retrying... ({retries}/{maxRetries} retries){Fore.RESET}")
                else: # If proposal is not found, check if one is required 
                    # Agents are allowed to not have a proposal if initial allocation is not required
                    if (self.hasInitialProposal and self.numIterations == 0): # if we need an initial allocation in the first round
                        print(f"{Fore.RED}Invalid proposal: Proposal Not Found In Initial Message{Fore.RESET}")
                        helperMessage = f"IMPORTANT: Just this once, you must propose the given initial proposal with the 'PROPSAL:' keyword. Remember, the initial proposal is:\n{self.agent1.agentName}: {', '.join([task.mappedName for task in self.initialProposal.agent1Tasks])}\n{self.agent2.agentName}: {', '.join([task.mappedName for task in self.initialProposal.agent2Tasks])}\nTry Again"
                        currentAgent.addToChatHistory('system', helperMessage)
                        retries += 1
                        print(f"{Fore.RED}Retrying... ({retries}/{maxRetries} retries){Fore.RESET}")
                    else:
                        break
                    
            # check for DNF
            if retries >= maxRetries:
                print(f"{Fore.RED}Negotiation Did Not Finish: {currentAgent.agentName} could not produce a valid proposal.{Fore.RESET}")
                self.DNF = True
                break
            
            if potentialProposal.hasDeal: # Check if the agent has accepted the proposal
                dealCounter += 1
                if dealCounter == 2:
                    agreementReached = True
            else: # If the agent has not accepted the proposal, reset the deal counter
                dealCounter = 0

            print(f"\n{Fore.CYAN}{currentAgent.agentName}:{Fore.RESET}\n{currentResponse}")
            
            # print(f"\n\n{currentAgent.agentName}'s Memory:\n")
            # currentAgent.printMemory()
            
            # Prepare for the next iteration
            currentAgent, otherAgent = otherAgent, currentAgent # Switch agents
            currentInput = currentResponse
            self.numIterations += 1
        
        if self.maxIterations == self.numIterations:
            print(f"{Fore.RED}Negotiation Did Not Finish: Max Iterations Reached{Fore.RESET}")
            self.DNF = True
        
        negotiationEndTime = datetime.datetime.now().replace(microsecond=0)
        self.negotiationTime = negotiationEndTime - negotiationStartTime
        self.winningProposal = self.findMostRecentProposal(currentAgent)
        
    def doesProposalMatchInitialProposal(self, proposal):
        """
        Checks if the proposal matches the initial allocation. Returns True if it does, False otherwise.
        """
        proposalGroup1 = set(proposal.agent1Tasks)
        proposalGroup2 = set(proposal.agent2Tasks)
        initialGroup1 = set(self.initialProposal.agent1Tasks)
        initialGroup2 = set(self.initialProposal.agent2Tasks)
        return (proposalGroup1 == initialGroup1 and proposalGroup2 == initialGroup2)
            
    def extractProposalFromReponse(self, response):
        """
        Extracts a Proposal object from a response string. Assumes the response
        contains a proposal in the new formal format:
                
        formal_proposal = {
            "Joe": ["Running", "Fishing"],
            "Fred": ["Hiking", "Cycling"],
            "has_deal": "True"
        }
        """

        # Look for the new formal proposal keyword
        if "formal_proposal" not in response:
            return NegotiationFlag.PROPOSAL_NOT_FOUND

        # Extract the dictionary part from the response
        # Extract exactly the 5 lines of the proposal, starting from the line with the opening curly brace
        lines = response.splitlines()
        proposal_str = ""
        for i, line in enumerate(lines):
            if "formal_proposal" in line:
                proposal_str = "\n".join(lines[i:i+5])
                break
            
        try:
            dict_start = proposal_str.index('{')
            dict_end = proposal_str.rindex('}')
            dict_substring = proposal_str[dict_start:dict_end + 1]
            proposal_dict = ast.literal_eval(dict_substring) # ex: proposal_dict= {'Finn_items': ['Painting', 'Puzzle'], 'Jake_items': ['Chess', 'Piano', 'Gardening', 'Guitar'], 'has_deal': 'False'}
        except Exception:
            return NegotiationFlag.INVALID_PROPOSAL_FORMAT

        agent1Tasks = []
        agent2Tasks = []

        # Extract tasks for agent1 if present
        agent1Key = self.agent1.agentName + "_items"
        if agent1Key in proposal_dict:
            for taskName in proposal_dict[agent1Key]:
                task = self.convertTaskNameToTask(taskName)
                agent1Tasks.append(task)

        agent2Key = self.agent2.agentName + "_items"
        # Extract tasks for agent2 if present
        if agent2Key in proposal_dict:
            for taskName in proposal_dict[agent2Key]:
                task = self.convertTaskNameToTask(taskName)
                agent2Tasks.append(task)

        # Extract has_deal boolean
        has_deal = proposal_dict.get('has_deal', 'False').lower() == 'true'

        return Proposal(agent1Tasks, agent2Tasks, has_deal)
    
    def extractTasksFromLine(self, line):
        tasks = []        
        tasksPart = line.split(":", 1)[1].strip()
        tasksList = [task.strip() for task in tasksPart.split(",") if task.strip()]
        for taskName in tasksList:
            task = self.convertTaskNameToTask(taskName)
            tasks.append(task)
        return tasks
        
    def convertTaskNameToTask(self, taskName):
        for task in self.tasks:
            if task.mappedName.upper() == taskName.upper():
                return task
        return Task(name=taskName, pref1=0.0, pref2=0.0) # Return a dummy task if the task name is not found
    
    def findMostRecentProposal(self, currentAgent):
        # Traverse the currentAgent's memory in reverse order to find the most recent proposal
        for message in reversed(currentAgent.memory):
            if (isinstance(message, AIMessage) or isinstance(message, HumanMessage)) \
            and 'formal_proposal' in message.content:
                return self.extractProposalFromReponse(message.content)
        # If no proposal was found, return None or raise an error.
        print("Warning: No recent proposal found in agent memory.")
        return None

        