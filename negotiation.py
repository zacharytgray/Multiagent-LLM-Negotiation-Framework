from agent import Agent
from task import Task
from colorama import Fore
from proposal import Proposal
from negotiationFlag import NegotiationFlag
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from negotiationManager import NegotiationManager
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
            "YOUR RESPONSE DID NOT INCLUDE A JSON OBJECT WITH THE PROPOSED TASKS."
            "Reformat your new response to include the proposed tasks in the JSON format."
            # f"{self.formattingReminder}"
        )
        self.invalidJSONKeyWarning = (
            "\n\n**ERROR: INVALID JSON KEY**\n"
            "YOUR RESPONSE INCLUDED AT LEAST ONE INVALID JSON KEY."
            "Remember to include the JSON object with the proposed tasks in your response."
            "Please ensure that the JSON object includes the keys 'my_tasks', 'partner_tasks', and 'has_deal'."
        )
        
    def updateAgentInstructions(self): # Add the negotiation tasks to the agent's instructions
        self.agent1.systemInstructions += "\n**NOW, HERE ARE THE ACTUAL ITEMS YOU MUST ALLOCATE:**\n\n"
        self.agent2.systemInstructions += "\n**NOW, HERE ARE THE ACTUAL ITEMS YOU MUST ALLOCATE:**\n\n"
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
        manager = NegotiationManager(self)
        negotiation_start_time, current_agent, other_agent = manager.initialize_negotiation()
        
        current_input = f"Hello, I am {other_agent.agentName}. Let's begin the task negotiation. Please start the negotiation process."
        
        while not manager.agreement_reached and self.numIterations < self.maxIterations:
            if self.numIterations <= 1 and self.hasInitialProposal:
                self.updateAgentInitialInstructions(current_agent, other_agent)
                
            current_response, proposal = manager.process_proposal(current_agent, other_agent, current_input)
            
            if current_response is None:
                print(f"{Fore.RED}Negotiation Did Not Finish: {current_agent.agentName} could not produce a valid proposal.{Fore.RESET}")
                self.DNF = True
                break
                
            if proposal.hasDeal:
                manager.deal_counter += 1
                if manager.deal_counter == 2:
                    manager.agreement_reached = True
            else:
                manager.deal_counter = 0

            print(f"\n{Fore.CYAN}{current_agent.agentName}:{Fore.RESET}\n{current_response}")
            
            self.numIterations += 1
            self.clearAllSystemMessages(current_agent)
            current_agent, other_agent = other_agent, current_agent
            current_input = current_response
            
        if self.maxIterations == self.numIterations:
            print(f"{Fore.RED}Negotiation Did Not Finish: Max Iterations Reached{Fore.RESET}")
            self.DNF = True
        
        negotiation_end_time = datetime.datetime.now().replace(microsecond=0)
        self.negotiationTime = negotiation_end_time - negotiation_start_time
        self.winningProposal = self.findMostRecentProposal(other_agent)

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
        
        helperMessage = f"""{self.missingProposalWarning}\n
Remember, for this round, set your proposal to the following initial allocation:

json
{{
    '{agent1Key}': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent1Tasks])}],
    '{agent2Key}': [{', '.join([f'"{task.mappedName}"' for task in self.initialProposal.agent2Tasks])}],
    'has_deal': 'False'
}}"""
        return helperMessage   
    def clearAllSystemMessages(self, agent):
        # Keep only first system message and non-system messages
        agent.memory = [agent.memory[0]] + [msg for msg in agent.memory[1:] if not isinstance(msg, SystemMessage)]