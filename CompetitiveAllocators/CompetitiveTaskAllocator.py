import os
import ast
import ollama
from colorama import Fore
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI 
from langchain_ollama import ChatOllama 
import itertools

load_dotenv("/Users/zacharytgray/Documents/GitHub/Ollama-LLM-Sandbox/keys.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
	print("API key for OpenAI not found.")
client = ChatOpenAI(openai_api_key=openai_api_key)  # Updated initialization

def logAssignedItems(fileName, boardState):
	agent1Name = boardState.agent1.name
	agent2Name = boardState.agent2.name
	agent1Items = boardState.getItems(agent1Name)
	agent2Items = boardState.getItems(agent2Name)
	with open(fileName, "a") as f:
		f.write(f"\n{agent1Name}'s items: {agent1Items}\n")
		f.write(f"{agent2Name}'s items: {agent2Items}\n")

class Agent:
	def __init__(self, name, model_name, systemInstructionsFilename, useOpenAI=False):
		self.name = name
		self.model_name = model_name
		self.numTokensGenerated = 0
		self.memory = []  # Replace ConversationBufferMemory with a simple list
		self.useOpenAI = useOpenAI
		self.systemInstructions = f"Your name is {self.name}. "
		self.instructionsFilename = systemInstructionsFilename

		if useOpenAI:
			if openai_api_key is None:
				print("OpenAI API key is not set.")
				exit(1)
			self.model = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)  # Updated parameter
		else:
			self.model = ChatOllama(model=model_name, base_url="http://localhost:11434")

		if systemInstructionsFilename != "":
			try:
				with open(self.instructionsFilename, "r") as f:
					self.systemInstructions += f.read()
				self.addToMemoryBuffer('system', self.systemInstructions)
			except FileNotFoundError:
				print(f"Error: {self.instructionsFilename} not found.")
				exit(1)

	def addToMemoryBuffer(self, role, content):
		if role == 'system':
			self.memory.append(SystemMessage(content=content))
		elif role == 'user':
			self.memory.append(HumanMessage(content=content))
		elif role == 'assistant':
			self.memory.append(AIMessage(content=content))
		else:
			raise ValueError(f"Unknown role: {role}")

	def run(self, role, inputText):
		self.addToMemoryBuffer(role, inputText)
		prompt = ChatPromptTemplate.from_messages(self.memory)
		chain = prompt | self.model
		response = chain.invoke({})
		# Extract the content from the response
		response_content = response.content if isinstance(response, AIMessage) else response
		self.addToMemoryBuffer('assistant', response_content)
		return response_content.strip()

class BoulwareAgent(Agent):
	def __init__(self, name, model_name, systemInstructionsFilename, items, useOpenAI=False):
		super().__init__(name, model_name, systemInstructionsFilename, useOpenAI)
		self.items = items  # Store items as an instance attribute
		self.offers_made = 0  # Count the number of offers made
		self.max_offers = 5   # Maximum number of offers before conceding
		self.utility_rankings = []  # Ranked list of allocations
		self.best_allocation = None

	def getAllocationRankings(self, items):
		# Rank all possible allocations based on Agent 2's preferences. Index 0 should be the best allocation, and the last index should be the worst.
		self.utility_rankings = []

		# Consider all possible combinations of items
		for i in range(1, len(items)):
			all_allocations = list(itertools.combinations(items, i))
			for group1 in all_allocations:
				group2 = [item for item in items if item not in group1]
				utility1 = sum(item.pref2 for item in group1)
				utility2 = sum(item.pref2 for item in group2)
				self.utility_rankings.append((group1, group2, utility1 + utility2))

		# Sort allocations by the combined utility in descending order
		self.utility_rankings.sort(key=lambda x: x[2], reverse=True)
		self.best_allocation = self.utility_rankings[0]
		
	def find_proposal_index(self, proposal):
		for index, (group1, group2, _) in enumerate(self.utility_rankings):
			if set(item.name for item in group1) == set(proposal):
				return index
		return len(self.utility_rankings) - 1  # If not found, assume worst offer
	
	def generate_boulware_offer(self):
		# Generate an offer dictionary that assigns all items to agents
		self.getAllocationRankings(self.items)  # Use self.items instead of boardState.items
		offer = {}
		best_allocation = self.best_allocation  # (group1, group2, utility)
		group1, group2, _ = best_allocation
		for item in group1:
			offer[item.name] = self.name
		for item in group2:
			offer[item.name] = 'Agent 1' if self.name == 'Agent 2' else 'Agent 2'
		return offer # formatted as {'item1':'Agent 1', 'item2':'Agent 2', ...}

	def should_accept(self, proposal):
		# Determine whether to accept the proposal
		proposal_index = self.find_proposal_index(proposal)
		if proposal_index <= self.offers_made:
			return True
		return False

	def run(self, role, inputText):
		self.addToMemoryBuffer(role, inputText)
		# Parse the last proposal from the other agent
		potential_proposal = self.parse_proposal(inputText)
  
		# Check if the proposal is not empty and is complete (all items are allocated)
		if potential_proposal and len(potential_proposal) == len(self.items):
			last_proposal = potential_proposal
			print(f"Last proposal: {last_proposal}")

			# Decide whether to accept or propose a new offer
			if self.should_accept(last_proposal):
				response_content = "I accept your offer."
				# Signal to call the moderator if needed
			else:
				# Propose a new offer based on Boulware strategy
				# new_offer = self.generate_boulware_offer()
				new_offer = self.best_allocation
				response_content = f"I propose the following allocation: {new_offer}"
				self.offers_made += 1
		else:
			# No valid proposal found, proceed with initial or new offer
			new_offer = self.generate_boulware_offer()
			response_content = f"I propose the following allocation: {new_offer}"
			self.offers_made += 1

		self.addToMemoryBuffer('assistant', response_content)
		return response_content.strip()

	def parse_proposal(self, inputText):
		# Determine the other agent's name
		other_agent_name = 'Agent 1' if self.name != 'Agent 1' else 'Agent 2'
		# Initialize the moderator agent if not already
		if not hasattr(self, 'moderatorAgent'):
			if self.useOpenAI:
				self.moderatorAgent = Agent("Moderator", self.model_name, "", useOpenAI=self.useOpenAI)
			else:
				self.moderatorAgent = Agent("Moderator", self.model_name, "", useOpenAI=self.useOpenAI)


		# Clear the moderator's memory
		self.moderatorAgent.memory = []	

		# Set system instructions similar to getConsensus
		self.moderatorAgent.systemInstructions = f"""
You have just received a message from {other_agent_name} in an item allocation negotiation.

Rules:
- Extract the proposed allocation from the message.
- Return the proposed allocation in a dictionary format: {{'Item A':'AGENT NAME', 'Item B':'AGENT NAME', ...}}
- Only include items that {other_agent_name} is proposing to assign to themselves.
- 'AGENT NAME' should be '{self.name}' or '{other_agent_name}'.
- Do not include any extra text, only return the dictionary.

If no proposal is found, return an empty dictionary.
		"""

		# Add the input text to the moderator's memory
		self.moderatorAgent.addToMemoryBuffer('user', inputText)

		# Run the moderator agent to parse the proposal
		isDict = False
		notDictErrorStr = "You did not return a valid dictionary. Please follow the format precisely, with no additional text."

		while not isDict:
			response = self.moderatorAgent.run('user', self.moderatorAgent.systemInstructions)
			try:
				proposal_dict = ast.literal_eval(response)
				if isinstance(proposal_dict, dict):
					isDict = True
				else:
					self.moderatorAgent.addToMemoryBuffer('user', notDictErrorStr)
			except (ValueError, SyntaxError):
				self.moderatorAgent.addToMemoryBuffer('user', notDictErrorStr)

		return proposal_dict

	def find_proposal_index(self, proposal):
		# Find the index of the proposal in the utility rankings
		for index, (group1, _, _) in enumerate(self.utility_rankings):
			proposal_items = set(item.name for item in group1)
			proposed_items = set(item for item, agent in proposal.items() if agent == self.name)
			if proposal_items == proposed_items:
				return index
		return len(self.utility_rankings) - 1  # If not found, assume worst offer

class Item:
	def __init__(self, name, pref1, pref2):
		self.name = name
		self.pref1 = pref1
		self.pref2 = pref2
	
	def __repr__(self):
		return f"{self.name} ({self.pref1}, {self.pref2})"
		
class BoardState:
    def __init__(self, agent1: Agent, agent2: Agent, items: list):
        self.agent1 = agent1
        self.agent2 = agent2
        self.items = items
        self.board = {
            self.agent1.name: [],
            self.agent2.name: []
        }

    def getItems(self, agent_name: str):
        return self.board[agent_name]

    def resetItems(self):
        self.board[self.agent1.name] = []
        self.board[self.agent2.name] = []

    def addItem(self, item: Item, agent_name: str):
        if item not in self.board[agent_name]:
            if item in self.items:
                # Remove item from the other agent if assigned
                other_agent = self.agent1.name if agent_name == self.agent2.name else self.agent2.name
                if item in self.board[other_agent]:
                    self.board[other_agent].remove(item)
                self.board[agent_name].append(item)
            else:
                print(f"Error: {item} not in item list.")
        else:
            print(f"Error: {item} already assigned to {agent_name}.")

    def getItemIndex(self, item_name: str):
        for i, item in enumerate(self.items):
            if item.name.lower().strip() == item_name.lower().strip():
                return i
        return None

class Domain:
	def __init__(self, items: list, agent1Model: str, agent2Model: str, moderatorModel: str, agent1UseOpenAI, agent2UseOpenAI, moderatorUseOpenAI):
		filePath = "CompetitiveAllocators/CompetitiveSystemInstructions.txt"
		self.items: list[Item] = items
		self.agent1 = Agent("Agent 1", agent1Model, filePath, useOpenAI=agent1UseOpenAI)
		self.agent2 = BoulwareAgent("Agent 2", agent2Model, filePath, items, useOpenAI=agent2UseOpenAI)
		self.moderatorAgent = Agent("Moderator", moderatorModel, "", useOpenAI=moderatorUseOpenAI)
		self.numItems = len(items)
		self.numConversationIterations = 0
		self.boardState = BoardState(self.agent1, self.agent2, items)
		self.consensusCounter = 0
  
		for item in self.items:
			self.agent1.systemInstructions +=  f"\n- {item.name}: {self.agent1.name}, Your preference value for this item is {item.pref1} out of 1.0"
			self.agent2.systemInstructions +=  f"\n- {item.name}: {self.agent2.name}, Your preference value for this item is {item.pref2} out of 1.0"
		
		self.agent1.systemInstructions += "\n\nLet's begin! Remember to be concise."
		self.agent2.systemInstructions += "\n\nLet's begin! Remember to be concise."
  
	def getItemIndex(self, item_name: str):
		for i, item in enumerate(self.items):
			if item.name.lower().strip() == item_name.lower().strip():
				return i
		return None

	def getConsensus(self, boardState):
		self.consensusCounter += 1
		boardState.resetItems()
		self.moderatorAgent.memory = []  # Reset moderator's memory

		if self.consensusCounter > 3:
			self.moderatorAgent.systemInstructions = f""" 
You have just received a conversation between {self.agent1.name} and {self.agent2.name}. You are the moderator for this item allocation negotiation.
These two partners have been asked to allocate {self.numItems} items between each other based on their own preferences for each item.
Rules:
- You are not going to change their decisions in any way. 
- Your job is to simply show me the results of their conversation in a dictionary format: {{'Item A':'AGENT NAME', 'Item B':'AGENT NAME', ...}}
- Return the results that they seem to agree on the most. If they do not agree on an item, make your best guess on the item assignment based on their conversation.
- Note that collaboration or splitting an item is strictly forbidden. If they want to collaborate on a task, make an educated guess as to who gets the item.
- Note that 'AGENT NAME' is a placeholder for the name of the agent you think should be assigned that item based on their conversation. It should be replaced with '{self.agent1.name}' or '{self.agent2.name}'.
- Include apostrophes as shown around the item names and agent names to ensure the dictionary is formatted correctly.
- To ensure your message is a valid dictionary, make sure your response starts with an open curly bracket and ends with a curly bracket. 
- Do not inculde leading or trailing apostrophes. 
- Do not inclue any headers like 'python' or 'json'.
- Do not respond with any extra text, not even an introduction. Simply return the following, replacing 'AGENT NAME' as needed:
   

"""
		else:
			self.moderatorAgent.systemInstructions = f"""
You have just received a conversation between {self.agent1.name} and {self.agent2.name}. You are the moderator for this item allocation negotiation.
These two partners have been asked to allocate {self.numItems} items between each other based on their own preferences for each item.
Rules:
- You are not going to change their decisions in any way. 
- Your job is to simply show me the results of their conversation in a dictionary format: {{'Item A':'AGENT NAME', 'Item B':'AGENT NAME', ...}}
- Return the most recent allocation discussed that they seem to agree on the most. 
- If they do not yet agree on an item, return 'TBD' for that item.
- If they decide to collaborate or split an item, return 'COLLAB' for that item.
- Note that 'AGENT NAME' is a placeholder for the name of the agent you think should be assigned that item based on their conversation. It should be replaced with '{self.agent1.name}', '{self.agent2.name}', or 'TBD' if they have not come to a consensus on that item.
- Include apostrophes as shown around the item names and agent names to ensure the dictionary is formatted correctly.
- To ensure your message is a valid dictionary, make sure your response starts with an open curly bracket and ends with a curly bracket. 
- Do not inculde leading or trailing apostrophes. 
- Do not inclue any headers like 'python' or 'json'.
- **DO NOT COME UP WITH YOUR OWN ALLOCATION. ONLY RETURN THE ALLOCATION THAT THEY HAVE AGREED ON.**
- Do not respond with any extra text, not even an introduction. Simply return the following, replacing 'AGENT NAME' as needed:

"""

		self.moderatorAgent.systemInstructions += "{"
		for item in self.items:
			self.moderatorAgent.systemInstructions += f"'{item.name}':'AGENT NAME'," 
		self.moderatorAgent.systemInstructions = self.moderatorAgent.systemInstructions[:-1]
		self.moderatorAgent.systemInstructions += "}"
   
		memory = self.agent1.memory 
		for message in memory:
			if isinstance(message, HumanMessage):
				self.moderatorAgent.addToMemoryBuffer('user', f"{self.agent2.name}'s Response: {message.content}")
			elif isinstance(message, AIMessage):
				self.moderatorAgent.addToMemoryBuffer('user', f"{self.agent1.name}'s Response: {message.content}")

		isDict = False
		notDictErrorStr = "You did not return a valid dictionary. Please follow the format precisely, with no additional text. A dictionary is formatted as {'key1':'value1', 'key2':'value2', ...}."
		while not isDict:
			rawConsensus = self.moderatorAgent.run('user', self.moderatorAgent.systemInstructions) # Should be {'item name':'agent name', ...}
			try:
				consensusDict = ast.literal_eval(rawConsensus)
				if not isinstance(consensusDict, dict):
					print("Error: rawConsensus is not a dictionary.")
				else:
					isDict = True
			except (ValueError, SyntaxError):
				print("Error: rawConsensus is not a valid Python dictionary.")	
			if not isDict:
				print("Raw consensus:\n	" + rawConsensus)
				self.moderatorAgent.addToMemoryBuffer('user', notDictErrorStr)

				
		disagreedItems = "" # str of items that the agents disagreed on
		index = 0
		for rawitem, agent in consensusDict.items():
			if not rawitem or not agent:
				print(f"{Fore.RED}Error: Empty item or agent name in consensus: {rawitem}, {agent}{Fore.RESET}")
				print(f"Raw Consensus: \n{rawConsensus}{Fore.RESET}")
				return False

			assignedItem = rawitem.lower().strip()
			assignedAgent = agent.lower().strip()
   
			assignedItemInItems = False
			for item in boardState.items: # Check if item name is valid
				# Debug print:
				if assignedItem == item.name.lower().strip():
					assignedItemInItems = True
 
			if not assignedItemInItems:
				print(f"{Fore.RED}Error: Invalid item name in consensus: {assignedItem}{Fore.RESET}")
				print(f"Raw Consensus: \n{rawConsensus}{Fore.RESET}")
				return False
			
			if index < self.numItems:
				if assignedAgent == self.agent1.name.lower().strip():
					itemidx = self.getItemIndex(assignedItem)
					boardState.addItem(self.items[itemidx], self.agent1.name)
				elif assignedAgent == self.agent2.name.lower().strip():
					itemidx = self.getItemIndex(assignedItem)
					boardState.addItem(self.items[itemidx], self.agent2.name)
				elif assignedAgent == "tbd":
					itemidx = self.getItemIndex(assignedItem)
					print(f"No consensus reached for {self.items[itemidx].name}.")
					disagreedItems += assignedItem + ", "
				elif assignedAgent == "collab":
					itemidx = self.getItemIndex(assignedItem)
					self.agent1.addToMemoryBuffer('user', f"You and {self.agent2.name} have incorrectly decided to collaborate on or split item {self.items[itemidx].name}. Splitting an item is not allowed. Please reevaluate your decisions conversationally, assigning each item to just one person. Ensure every item gets allocted.")
					self.agent2.addToMemoryBuffer('user', f"You and {self.agent1.name} have incorrectly decided to collaborate on or split item {self.items[itemidx].name}. Splitting an item is not allowed. Please reevaluate your decisions conversationally, assigning each item to just one person. Ensure every item gets allocted.")
					return False
				else:
					print(f"{Fore.RED}Error: Invalid agent name in consensus: {assignedAgent} not equal to {self.agent1.name.lower().strip()} or {self.agent2.name.lower().strip()}{Fore.RESET}")
					print(f"Raw Consensus: \n{rawConsensus}{Fore.RESET}")
					return False
			index += 1

		if disagreedItems != "":
			print(f"{Fore.RED}Disagreed on items: {disagreedItems}{Fore.RESET}")
			self.agent1.addToMemoryBuffer('user', f"You did not come to complete agreement with {self.agent2.name} on item(s) {disagreedItems[:-1]}. Please continue discussion to finalize the allocation. You must allocate all the items.")
			self.agent2.addToMemoryBuffer('user', f"You did not come to complete agreement with {self.agent1.name} on item(s) {disagreedItems[:-1]}. Please continue discussion to finalize the allocation. You must allocate all the items.")
			return False

		total_assigned_items = len(boardState.getItems(self.agent1.name)) + len(boardState.getItems(self.agent2.name))
		if total_assigned_items > self.numItems:
			print(f"{Fore.RED}Error: Too many items assigned.")
			print(f"{self.agent1.name}'s items: {boardState.getItems(self.agent1.name)}")
			print(f"{self.agent2.name}'s items: {boardState.getItems(self.agent2.name)}{Fore.RESET}")
			return False
		elif total_assigned_items < self.numItems:
			print(f"{Fore.RED}Error: Not all items assigned.")
			print(f"{self.agent1.name}'s items: {boardState.getItems(self.agent1.name)}")
			print(f"{self.agent2.name}'s items: {boardState.getItems(self.agent2.name)}{Fore.RESET}")
			return False
  
		return True

	def printItems(self):
		print(f"\n{Fore.YELLOW}{self.agent1.name}'s items: {self.boardState.getItems(self.agent1.name)}")
		print(f"{self.agent2.name}'s items: {self.boardState.getItems(self.agent2.name)}{Fore.RESET}")

	def startNegotiation(self, numIterations, startingAgent):
		self.agent1.addToMemoryBuffer('system', self.agent1.systemInstructions)
		self.agent2.addToMemoryBuffer('system', self.agent2.systemInstructions)

		if startingAgent == self.agent1:
			currentAgent = self.agent1
			otherAgent = self.agent2
		else:
			currentAgent = self.agent2
			otherAgent = self.agent1
		
		currentInput = f"Hello! I'm {otherAgent.name}. Let's begin the item allocation negotiation. Please start the negotiation process."
		otherAgent.addToMemoryBuffer('assistant', currentInput)
	
		consensusReached = False

		while not consensusReached:

			for i in range(numIterations):
				response = currentAgent.run("user", currentInput)
				if currentAgent == self.agent1:
					print(f"{Fore.LIGHTBLUE_EX}\n{currentAgent.name}: \n	{response}{Fore.RESET}")
				elif currentAgent == self.agent2:
					print(f"{Fore.LIGHTMAGENTA_EX}\n{currentAgent.name}: \n	{response}{Fore.RESET}")

				currentAgent = self.agent2 if currentAgent == self.agent1 else self.agent1
				currentInput = response
				self.numConversationIterations += 1

				if i == (numIterations-1): # Manually add final dialogue to agent 2
					self.agent2.addToMemoryBuffer('user', currentInput)
	
			print(f"\nAsking Moderator for Consensus...")
   
			consensusReached = self.getConsensus(self.boardState)
   
if __name__ == '__main__':
	pass  # Add code to initialize and start domain negotiation if necessary