from agent import Agent
from task import Task
from colorama import Fore
from proposal import Proposal
from negotiationFlag import NegotiationFlag
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import random
import datetime
import copy
import ast
import json

class Negotiation:
    def __init__(self, roundIndex, numTasks, maxIterations, agent1Model, agent1usesOpenAI, agent1Type, agent2Model, agent2usesOpenAI, agent2Type, agent1Name, agent2Name, hasInitialProposal):
        self.roundIndex = roundIndex
        self.DNF = False # Did Not Finish
        self.seed = str(roundIndex) + " I love LLMs!" # Seed for random number generation
        self.numTasks = numTasks
        self.agent1 = Agent(agentName=agent1Name, modelName=agent1Model, usesOpenAI=agent1usesOpenAI, agentType=agent1Type)
        self.agent2 = Agent(agentName=agent2Name, modelName=agent2Model, usesOpenAI=agent2usesOpenAI, agentType=agent2Type)
        self.numIterations = 0 # Number of conversation iterations in the negotiation
        self.maxIterations = maxIterations
        self.hasInitialProposal = hasInitialProposal
        self.initialProposal = None # The initial proposal, set in setUpInitialProposal()
        # self.tasks = self.generateTasks(self.numTasks) # or use generateRandomTasks() for random tasks
        self.tasks = self.generateRandomTasks(self.numTasks)
        self.tasks = self.generateSeededTasks(self.numTasks)
        self.negotiationTime = 0 # Time taken for the negotiation
        self.winningProposal = None # The winning proposal at the end of the negotiation
        self.formattingReminder = self.setFormattingReminder() # Initiate the formatting reminder
        self.proposalFormatExample = None # Initiate the proposal formatting example
        self.missingProposalWarning = (
            "\n\n**ERROR: MISSING PROPOSAL**\n"
            "YOUR RESPONSE DID NOT INCLUDE A PROPOSAL.\n"
            "You MUST include a proposal in every response.\n"
            f"Please format your proposal as follows:{self.formattingReminder}"
        )
        
    def updateAgentInstructions(self): # Add the negotiation tasks to the agent's instructions
        self.agent1.systemInstructions += "\n**NOW, HERE ARE THE ACTUAL ITEMS YOU MUST ALLOCATE:**\n"
        self.agent2.systemInstructions += "\n**NOW, HERE ARE THE ACTUAL ITEMS YOU MUST ALLOCATE:**\n"
        for task in self.tasks:
            self.agent1.systemInstructions += f"{task.mappedName}: Your confidence level for this task is {task.confidence1}.\n"
            self.agent2.systemInstructions += f"{task.mappedName}: Your confidence level for this task is {task.confidence2}.\n"
        
        # Replace agent names in system instructions 
        self.agent1.systemInstructions = self.agent1.systemInstructions.replace("[my_name]", self.agent1.agentName).replace("[partner_name]", self.agent2.agentName)
        self.agent2.systemInstructions = self.agent2.systemInstructions.replace("[my_name]", self.agent2.agentName).replace("[partner_name]", self.agent1.agentName)
        
        # Replace agent names and items in initial proposal helper instructions
        if self.hasInitialProposal:
            self.agent1.initialProposalHelperInstructions = self.agent1.initialProposalHelperInstructions.replace("[my_name]", self.agent1.agentName).replace("[partner_name]", self.agent2.agentName).replace("[myTasks]", ", ".join([f'"{task.mappedName}"' for task in self.initialProposal.agent1Tasks])).replace("[partnerTasks]", ", ".join([f'"{task.mappedName}"' for task in self.initialProposal.agent2Tasks]))
            self.agent2.initialProposalHelperInstructions = self.agent2.initialProposalHelperInstructions.replace("[my_name]", self.agent2.agentName).replace("[partner_name]", self.agent1.agentName).replace("[myTasks]", ", ".join([f'"{task.mappedName}"' for task in self.initialProposal.agent2Tasks])).replace("[partnerTasks]", ", ".join([f'"{task.mappedName}"' for task in self.initialProposal.agent1Tasks]))
            
    def setUpInitialProposal(self): # Randomly assign tasks to each agent
        shuffledTasks = copy.deepcopy(self.tasks)
        random.shuffle(shuffledTasks)
        agent1Tasks = shuffledTasks[:self.numTasks//2]
        agent2Tasks = shuffledTasks[self.numTasks//2:self.numTasks]
        self.initialProposal = Proposal(agent1Tasks, agent2Tasks)
        
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
    
    def generateSeededTasks(self, numTasks):
        random.seed(self.seed)
        tasks = []
        for i in range(numTasks):
            taskName = chr(65 + i) # A, B, C, ...
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
        
        self.setUpInitialProposal() # Set up the self.initialProposal
        self.proposalFormatExample = self.setProposalFormattingExample(currentAgent) # Set up the proposal formatting example
        self.updateAgentInstructions() # Add the negotiation tasks to the agent's instructions
        self.agent1.addToChatHistory('system', self.agent1.systemInstructions)
        self.agent2.addToChatHistory('system', self.agent2.systemInstructions)
            
        currentInput = f"Hello, I am {otherAgent.agentName}. Let's begin the task negotiation. Please start the negotiation process."
        agreementReached = False
        dealCounter = 0
        
        # print tasks for this negotiation
        print("Tasks for this negotiation:")
        for task in self.tasks:
            print(f"{task.mappedName}: \n     {self.agent1.agentName}: {task.confidence1}, \n     {self.agent2.agentName}: {task.confidence2}")
            
        while not agreementReached and self.numIterations < self.maxIterations:
            maxRetries = 5
            retries = 0
            
            if currentAgent.agentName == self.agent1.agentName:
                agent1Key = "my_tasks"
                agent2Key = "partner_tasks"
            else:
                agent1Key = "partner_tasks"
                agent2Key = "my_tasks"
            
            if self.numIterations <= 1 and self.hasInitialProposal:
                self.updateAgentInitialInstructions(currentAgent, otherAgent) # Update the agent's instructions for the current iteration
                
            while retries < maxRetries: # Keep retrying if the proposal is invalid  
                if retries == 0: # Generate the first response
                    currentResponse = currentAgent.generateResponse(role='user', inputText=currentInput)
                else: # Retry if the proposal is invalid, not inputting otherAgent input this time
                    # Remove second to last item from currentAgent.memory if it is a HumanMessage
                    if len(currentAgent.memory) >= 2 and isinstance(currentAgent.memory[-2], AIMessage):
                        currentAgent.memory.pop(-2) # Remove the last AIMessage from memory
                    currentResponse = currentAgent.generateResponse()
                if currentResponse == NegotiationFlag.TIMEOUTERROR:
                    print(f"{Fore.RED}Response Timeout{Fore.RESET}")
                    retries += 1
                    print(f"{Fore.RED}Retrying... ({retries}/{maxRetries} retries){Fore.RESET}")
                    continue # Skip to the next iteration of the while loop
                potentialProposal = self.extractProposalFromReponse(currentResponse, currentAgent)
                if potentialProposal != NegotiationFlag.PROPOSAL_NOT_FOUND \
                    and potentialProposal != NegotiationFlag.INVALID_PROPOSAL_FORMAT \
                        and potentialProposal != NegotiationFlag.INVALID_AGENT_NAME: # If proposal is found, check if it is valid
                    if (self.numIterations == 0):
                        if self.hasInitialProposal: 
                            if not self.doesProposalMatchInitialProposal(potentialProposal):
                                # if we want an initial allocation and it's the first round and proposal does not match initial proposal,
                                # reprompt agent for proposal, matching the initial proposal this time.
                                print(f"{Fore.RED}Invalid proposal: Does Not Match Initial Proposal{Fore.RESET}")
                                helperMessage = self.setHelperMessage(currentAgent)
                                self.clearAllSystemMessages(currentAgent)
                                currentAgent.addToChatHistory('system', helperMessage) 
                            else:
                                currentAgent.proposal = potentialProposal
                                break                            
                        elif potentialProposal.hasDeal:
                            print(f"{Fore.RED}Invalid proposal: Initial Proposal Cannot Be Accepted{Fore.RESET}")
                            helperMessage = f"IMPORTANT: YOU ENTERED AN INVALID PROPOSAL. You are not allowed to agree to the initial proposal (set 'has_deal' to 'False').\nTry Again"
                            helperMessage += self.formattingReminder
                            self.clearAllSystemMessages(currentAgent)
                            currentAgent.addToChatHistory('system', helperMessage)
                        else: # If proposal is valid, update the current agent's proposal
                            currentAgent.proposal = potentialProposal
                            break # Exit the retry loop when a valid proposal is found
                    else:  
                        isValidProposal = potentialProposal.validateProposal(self.tasks)
                        if isValidProposal == NegotiationFlag.ERROR_FREE: # If proposal is valid, update the current agent's proposal
                            currentAgent.proposal = potentialProposal # Update currentAgent.proposal
                            break # Exit the retry loop when a valid proposal is found
                        else: # If proposal is invalid, prompt the agent to correct it
                            # Remove the last incorrect response from the memory
                            if isValidProposal == NegotiationFlag.NOT_ENOUGH_TASKS:
                                # reprompt agent for proposal, including all tasks this time.
                                print(f"{Fore.RED}Invalid proposal: Not Enough Tasks{Fore.RESET}")
                                helperMessage = f"IMPORTANT: You must include all {len(self.tasks)} tasks in your proposal. Remember, these are the tasks for this negotiation: {', '.join([task.mappedName for task in self.tasks])}\nTry Again"
                                helperMessage += self.formattingReminder
                                self.clearAllSystemMessages(currentAgent)
                                currentAgent.addToChatHistory('system', helperMessage)
                            elif isValidProposal == NegotiationFlag.TOO_MANY_TASKS:
                                # reprompt agent for proposal, including only given tasks this time.
                                print(f"{Fore.RED}Invalid proposal: Too Many Tasks{Fore.RESET}")
                                helperMessage = f"IMPORTANT: You must include only {len(self.tasks)} tasks in your proposal. Remember, these are the tasks for this negotiation: {', '.join([task.mappedName for task in self.tasks])}.\nTry Again"
                                helperMessage += self.formattingReminder
                                self.clearAllSystemMessages(currentAgent)
                                currentAgent.addToChatHistory('system', helperMessage)
                            elif isValidProposal == NegotiationFlag.INVALID_TASKS_PRESENT:
                                # reprompt agent for proposal, including only given tasks this time.
                                print(f"{Fore.RED}Invalid proposal: Invalid Tasks Present{Fore.RESET}")
                                helperMessage = f"IMPORTANT: You must include only the following {len(self.tasks)} tasks in your proposal: {', '.join([task.mappedName for task in self.tasks])}.\nTry Again"
                                helperMessage += self.formattingReminder
                                self.clearAllSystemMessages(currentAgent)
                                currentAgent.addToChatHistory('system', helperMessage)      
                            else:
                                print(f"{Fore.RED}Invalid proposal: Unknown Error{Fore.RESET}")   
                                print(f"{Fore.RED}Proposal that caused error: \n{potentialProposal}{Fore.RESET}")   
                elif potentialProposal == NegotiationFlag.INVALID_PROPOSAL_FORMAT:
                    print(f"{Fore.RED}Invalid proposal: Invalid Proposal Format{Fore.RESET}")
                    helperMessage = f"""IMPORTANT: You must use the following JSON format with NO exceptions:
json
{{
    '{agent1Key}': ['Task A', 'Task B'],
    '{agent2Key}': ['Task C', 'Task D'],
    'has_deal': 'False'
}}
Replace the alphabetized task names with the actual task names you want to propose.
\nTry Again"""
                    self.clearAllSystemMessages(currentAgent)
                    currentAgent.addToChatHistory('system', helperMessage)
                elif potentialProposal == NegotiationFlag.INVALID_AGENT_NAME:
                    print(f"{Fore.RED}Invalid proposal: Invalid Agent Name{Fore.RESET}")
                    helperMessage = f"""IMPORTANT: You must use the following JSON format with NO exceptions. TAKE SPECIAL CARE TO WRITE THE NAMES CORRECTLY, exactly as shown below. {self.formattingReminder}\n\nTry Again"""
                    self.clearAllSystemMessages(currentAgent)
                    currentAgent.addToChatHistory('system', helperMessage)
                elif potentialProposal == NegotiationFlag.PROPOSAL_NOT_FOUND: # If proposal is not found, check if one is required 
                    if (self.hasInitialProposal and self.numIterations == 0): # if we need an initial allocation in the first round
                        print(f"{Fore.RED}Invalid proposal: Proposal Not Found In Initial Message{Fore.RESET}")
                        helperMessage = f"""IMPORTANT: YOU ENTERED AN INVALID PROPOSAL. Just this once, you must propose the following initial proposal exactly as follows: 
json
{{
    '{agent1Key}': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent1Tasks])}],
    '{agent2Key}': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent2Tasks])}],
    'has_deal': 'False'
    
}}\nMAKE SURE YOU ENTER THE TASKS EXACLTY AS SHOWN.\n\nTry Again"""
                        self.clearAllSystemMessages(currentAgent)
                        currentAgent.addToChatHistory('system', helperMessage)
                    else:
                        print(f"{Fore.RED}Invalid proposal: Proposal Not Found {currentAgent.agentName}: {currentResponse}{Fore.RESET}")
                        self.clearAllSystemMessages(currentAgent)
                        currentAgent.addToChatHistory('system', self.missingProposalWarning)
                retries += 1
                print(f"{Fore.RED}Retrying... ({retries}/{maxRetries} retries){Fore.RESET}")

            # check for DNF
            if retries >= maxRetries:
                print(f"{Fore.RED}Negotiation Did Not Finish: {currentAgent.agentName} could not produce a valid proposal.{Fore.RESET}")
                self.DNF = True
                currentAgent.printMemory()
                break
            
            if not isinstance(potentialProposal, NegotiationFlag) and potentialProposal.hasDeal: # Check if the agent has accepted the proposal
                dealCounter += 1
                if dealCounter == 2:
                    agreementReached = True
            else: # If the agent has not accepted the proposal, reset the deal counter
                dealCounter = 0

            print(f"\n{Fore.CYAN}{currentAgent.agentName}:{Fore.RESET}\n{currentResponse}")
            
            # Prepare for the next iteration
            self.numIterations += 1
            self.clearAllSystemMessages(currentAgent) # Clear all system warning messages from currentAgent's memory if there are any
            currentAgent, otherAgent = otherAgent, currentAgent # Switch agents
            currentInput = currentResponse
            
        if self.maxIterations == self.numIterations:
            print(f"{Fore.RED}Negotiation Did Not Finish: Max Iterations Reached{Fore.RESET}")
            self.DNF = True
        
        negotiationEndTime = datetime.datetime.now().replace(microsecond=0)
        self.negotiationTime = negotiationEndTime - negotiationStartTime
        self.winningProposal = self.findMostRecentProposal(otherAgent) # otherAgent because agents switched at the end, so the last proposal made was from otherAgent
        
    def doesProposalMatchInitialProposal(self, proposal):
        """
        Checks if the proposal matches the initial allocation. Returns True if it does, False otherwise.
        """
        proposalGroup1 = set(proposal.agent1Tasks)
        proposalGroup2 = set(proposal.agent2Tasks)
        initialGroup1 = set(self.initialProposal.agent1Tasks)
        initialGroup2 = set(self.initialProposal.agent2Tasks)
        return (proposalGroup1 == initialGroup1 and proposalGroup2 == initialGroup2)
            
    def extractProposalFromReponse(self, response, currentAgent):
        
        if "json" not in response or "has_deal" not in response:
            return NegotiationFlag.PROPOSAL_NOT_FOUND
        
        agent1Tasks = []
        agent2Tasks = []

        # Extract the dictionary part from the response
        lines = response.splitlines()
        proposal_str = ""
        in_json = False
        
        #Extract the JSON part from the response
        for line in lines:
            stripped = line.strip()
            if "json" in stripped:
                in_json = True
                continue
            if in_json and stripped:
                proposal_str += stripped
                if "}" in stripped:
                    break
                
        try:
            # Clean up the string
            proposal_str = proposal_str.replace('\n', '').replace(' ', '')                 
            try:
                proposal_dict = json.loads(proposal_str)
            except json.JSONDecodeError:
                print(f"JSON decode error: {e}")
                # Fallback to ast.literal_eval
                proposal_dict = ast.literal_eval(proposal_str)

            # Extract tasks for agent1 if present
            if currentAgent.agentName == self.agent1.agentName:
                agent1Key = "my_tasks"
                agent2Key = "partner_tasks"
            else:
                agent1Key = "partner_tasks"
                agent2Key = "my_tasks"
                
            # Extract tasks for agent1 if present
            if agent1Key in proposal_dict:
                for taskName in proposal_dict[agent1Key]:
                    task = self.convertTaskNameToTask(taskName)
                    agent1Tasks.append(task)
            else:
                # If agent1's tasks are not present, return INVALID_AGENT_NAME
                return NegotiationFlag.INVALID_AGENT_NAME
            
            # Extract tasks for agent2 if present
            if agent2Key in proposal_dict:
                for taskName in proposal_dict[agent2Key]:
                    task = self.convertTaskNameToTask(taskName)
                    agent2Tasks.append(task)
            else:
                # If agent2's tasks are not present, return INVALID_AGENT_NAME
                return NegotiationFlag.INVALID_AGENT_NAME

            # Extract has_deal boolean
            # has_deal = proposal_dict.get('has_deal', 'False').lower() == 'true'
            
            # Handle has_deal value - accept both string and boolean
            has_deal_value = proposal_dict.get('has_deal', False)
            if isinstance(has_deal_value, str):
                has_deal = has_deal_value.lower() == 'true'
            else:
                has_deal = bool(has_deal_value)
                
        except Exception as e:
            print(f"{Fore.RED}Parsing error: {e} {response}{Fore.RESET}")
            return NegotiationFlag.INVALID_PROPOSAL_FORMAT

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
            and 'json' in message.content:
                return self.extractProposalFromReponse(message.content, currentAgent)
        # If no proposal was found, return None or raise an error.
        print("Warning: No recent proposal found in agent memory.")
        return None

    def setFormattingReminder(self):
        # initiate the formatting reminder
        reminderStr = f"""
\n**FORMATTING REMINDER** 
When you make a proposal, start by formally proposing the your proposed allocation using the exact format described below, beginning with  'json':

... Text leading up to proposal ...

json
{{
    'my_tasks': [Task A, Task B, ...],
    'partner_tasks': [Task C, Task D, ...],
    'has_deal': 'False'
}}
THE PROPOSAL MUST BEGIN WITH 'json' AND BE IN THE EXACT FORMAT SHOWN ABOVE.
Use the exact JSON format as shown above, replacing the alphabetized task names with the actual tasks you think are best. 
Replace 'has_deal' with 'True' if you have come to a mutual agreement on the current proposal."""
        return reminderStr
    
    def setProposalFormattingExample(self, currentAgent):
        if currentAgent.agentName == self.agent1.agentName:
            agent1Key = "my_tasks"
            agent2Key = "partner_tasks"
        else:
            agent1Key = "partner_tasks"
            agent2Key = "my_tasks"
        
        exampleStr = f"""
\n\n**VERY IMPORTANT!** 
FOR THIS ROUND ONLY, You should start by formally proposing the following initial allocation using the JSON format with the 'json' keyword:
json
{{
    '{agent1Key}': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent1Tasks])}],
    '{agent2Key}': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent2Tasks])}],
    'has_deal': 'False'
}}\n
COPY THIS ALLOCATION EXACTLY, ESPECIALLY TASK ASSIGNMENTS, AND DO NOT CHANGE ANYTHING ELSE.
"""
        return exampleStr
    
    def updateAgentInitialInstructions(self, currentAgent, otherAgent):
        if self.numIterations == 0:
                # set currentAgent's 0th index in currentAgent.memory to currentAgent.initialProposalHelperInstructions
                print(f"{Fore.YELLOW}Switching {currentAgent.agentName}'s memory to initial proposal helper instructions{Fore.RESET}")
                currentAgent.memory[0] = SystemMessage(content=currentAgent.initialProposalHelperInstructions)
        elif self.numIterations == 1:
            # set currentAgent's 0th index in currentAgent.memory to currentAgent.systemInstructions
            print(f"{Fore.YELLOW}Switching {otherAgent.agentName}'s memory back to system instructions{Fore.RESET}")
            currentAgent.memory[0] = SystemMessage(content=currentAgent.systemInstructions)   
            
    def setHelperMessage(self, currentAgent):
        if currentAgent.agentName == self.agent1.agentName:
            agent1Key = "my_tasks"
            agent2Key = "partner_tasks"
        else:
            agent1Key = "partner_tasks"
            agent2Key = "my_tasks"
        helperMessage = f"""IMPORTANT: YOU ENTERED AN INVALID PROPOSAL. Just this once, you must propose the given initial proposal exactly as follows: 
json
{{
    '{agent1Key}': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent1Tasks])}],
    '{agent2Key}': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent2Tasks])}],
    'has_deal': 'False'
}}\nMAKE SURE YOU ENTER THE TASKS EXACLTY AS SHOWN.\n\nTry Again"""
        return helperMessage        

    def clearAllSystemMessages(self, agent):
        # Keep only first system message and non-system messages
        agent.memory = [agent.memory[0]] + [msg for msg in agent.memory[1:] if not isinstance(msg, SystemMessage)]