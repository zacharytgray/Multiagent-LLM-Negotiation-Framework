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
		"""Base run method for regular agents"""
		self.addToMemoryBuffer(role, inputText)
		prompt = ChatPromptTemplate.from_messages(self.memory)
		chain = prompt | self.model
		response = chain.invoke({})
		response_content = response.content if isinstance(response, AIMessage) else response
		self.addToMemoryBuffer('assistant', response_content)
		return response_content.strip()

class BoulwareStrategy:
	def __init__(self, items, agent1Name, agent2Name):
		self.items = items
		self.agent1Name = agent1Name
		self.agent2Name = agent2Name  # Add agent2Name
		self.currentProposal = None  # Agent 2's current planned proposal
		self.currentUtility = 0  # Utility of currentProposal
		self.currentProposalIndex = 0  # Index in utility rankings
		self.rankedAllocations = self.getAllocationRankings()
		self.NumRounds = 0  # Number of rounds passed

	def getAllocationRankings(self):
		# Generate all possible allocations grouped by utility, allowing variable group sizes
		allocations_by_utility = {}
		n = len(self.items)
		for k in range(1, n):  # Allow allocations with group sizes from 1 to n-1
			for comb in itertools.combinations(self.items, k):
				group1 = [item for item in self.items if item not in comb]  # Agent1's items
				group2 = comb  # Agent2's items (current agent)
				utility = sum(item.pref2 for item in group2)  # Agent2's utility
				# Store allocation with Agent1's items first, Agent2's items second
				allocation = (group1, group2, utility)
				if utility not in allocations_by_utility:
					allocations_by_utility[utility] = []
				allocations_by_utility[utility].append(allocation)

		# Sort the utilities in descending order (highest utility first)
		sorted_utilities = sorted(allocations_by_utility.keys(), reverse=True)
		ranked_allocations = [allocations_by_utility[utility] for utility in sorted_utilities]
		
		# Debug print to verify allocations are ranked correctly
		print("\nAgent 2's Allocation Rankings (Debug):")
		for tier_index, tier in enumerate(ranked_allocations):
			for alloc in tier:
				print(f"Tier {tier_index}: Agent 1: {[item.name for item in alloc[0]]}, "
					  f"Agent 2: {[item.name for item in alloc[1]]}, "
					  f"Utility: {alloc[2]}")
		
		return ranked_allocations

	def getProposedAllocation(self):
		# Return the currentProposal
		return self.currentProposal

	def shouldAccept(self, opponentOffer):
		# Find the tier index of opponent's offer
		opponentIndex = self.findProposalIndex(opponentOffer)
		# Should accept only if opponent's offer is in same or better tier (lower index)
		return opponentIndex >= 0 and opponentIndex <= self.currentProposalIndex

	def updateCurrentAllocation(self, numRounds):
		self.NumRounds = numRounds
		totalTiers = len(self.rankedAllocations)
		maxTierIndex = totalTiers - 1
		# Ensure that for the first round, currentTierIndex is 0
		if numRounds == 1:
			currentTierIndex = 0
		else:
			currentTierIndex = min(
				int(maxTierIndex * (1 - (0.9 ** (numRounds - 1)))), 
				maxTierIndex
			)
		self.currentProposalIndex = currentTierIndex
		self.currentProposal = self.rankedAllocations[currentTierIndex][0]  # Always select the top allocation in the tier

	def findProposalIndex(self, proposal):
		group1 = {item for item, agent in proposal.items() if agent == self.agent1Name}
		group2 = {item for item, agent in proposal.items() if agent == self.agent2Name}
		for tier_index, tier in enumerate(self.rankedAllocations):
			for alloc in tier:
				alloc_group1 = {item.name for item in alloc[0]}
				alloc_group2 = {item.name for item in alloc[1]}
				if alloc_group1 == group1 and alloc_group2 == group2:
					return tier_index
		print(f"Error: Proposal {proposal} not found in rankedAllocations.")
		return -1

class BoulwareAgent(Agent):
	def __init__(self, name, model_name, systemInstructionsFilename, items, agent1Name, useOpenAI=False):  # Added agent1Name parameter
		super().__init__(name, model_name, systemInstructionsFilename, useOpenAI)
		self.items = items
		self.strategy = BoulwareStrategy(items, agent1Name, self.name)  # Passed agent1Name and self.name to strategy
		self.opponentOffer = {}  # Opponent's last offer

	def run(self, role, inputText):
		"""Overridden run method for Boulware agents"""
		# First, add the input to memory
		self.addToMemoryBuffer(role, inputText)
		
		# Parse the opponent's proposal
		opponentUtility = 0
		opponentOffer = self.parse_proposal(inputText)
		if opponentOffer:  # Changed from != {} to proper boolean check
			self.opponentOffer = opponentOffer
			# Calculate utility of Agent 1's offer for Agent 2
			opponentUtility = sum(
				item.pref2 for item in self.items 
				if item.name in opponentOffer and opponentOffer[item.name] == self.name
			)

		# Update current proposal based on the number of rounds
		self.strategy.updateCurrentAllocation(self.strategy.NumRounds + 1)

		# Check if there is any allocation with utility >= opponentUtility
		can_propose_better = False
		for tier in self.strategy.rankedAllocations:
			for allocation in tier:
				if allocation[2] >= opponentUtility:
					can_propose_better = True
					break
			if can_propose_better:
				break

		if can_propose_better and not self.strategy.shouldAccept(self.opponentOffer):
			# Propose a better or equal allocation without prefixing agent names
			proposed_allocation = self.strategy.getProposedAllocation()
			# FIXED: Swapped group1_items and group2_items assignments to match the allocation
			group1_items = [item.name for item in proposed_allocation[0]]  # Agent 1's items
			group2_items = [item.name for item in proposed_allocation[1]]  # Agent 2's (self) items
			other_agent_name = 'Agent 1' if self.name != 'Agent 1' else 'Agent 2'
			systemMessage = (
				f"Propose the following allocation:\n"
				f"{other_agent_name}: {', '.join(group1_items)}\n"  # Give group1 items to Agent 1
				f"{self.name}: {', '.join(group2_items)}"          # Keep group2 items for self (Agent 2)
			)
			self.addToMemoryBuffer('system', systemMessage)
		else:
			 # Only accept if we should AND the offer is valid
			if self.opponentOffer and self.strategy.shouldAccept(self.opponentOffer):
				self.addToMemoryBuffer('assistant', "Deal!")
				return "Deal!"
			else:
				# Make counter-proposal instead of accepting
				proposed_allocation = self.strategy.getProposedAllocation()
				# ...rest of propose code...

		# Generate a response using the model
		prompt = ChatPromptTemplate.from_messages(self.memory)
		chain = prompt | self.model	
		response = chain.invoke({})
		response_content = response.content if isinstance(response, AIMessage) else response
		self.addToMemoryBuffer('assistant', response_content)

		# Print the current proposal index and opponent's offer index
		currentProposalIndex = self.strategy.currentProposalIndex
		opponentOfferIndex = self.strategy.findProposalIndex(self.opponentOffer)  # Updated reference
		print(f"Current Proposal Index: {currentProposalIndex}")
		print(f"Opponent's Offer Index: {opponentOfferIndex}")
  
		return response_content.strip()

	def parse_proposal(self, inputText):
		other_agent_name = 'Agent 1' if self.name != 'Agent 1' else 'Agent 2'
		if not hasattr(self, 'moderatorAgent'):
			if self.useOpenAI:
				self.moderatorAgent = Agent("Moderator", self.model_name, "", useOpenAI=self.useOpenAI)
			else:
				self.moderatorAgent = Agent("Moderator", self.model_name, "", useOpenAI=self.useOpenAI)

			# Make up to 3 attempts to parse the proposal
			max_attempts = 3
			for attempt in range(max_attempts):
				# Clear the moderator's memory for each attempt
				self.moderatorAgent.memory = []
				
				self.moderatorAgent.systemInstructions = f"""
				You have just received a message from {other_agent_name} in an item allocation negotiation.
				This is attempt {attempt + 1} of {max_attempts}.

				Rules:
				- Extract the proposed allocation from the message.
				- Return the proposed allocation in a dictionary format: {{'Item A':'AGENT NAME', 'Item B':'AGENT NAME', ...}}
				- 'AGENT NAME' must be either '{self.name}' or '{other_agent_name}'. Do NOT use pronouns like 'Me', 'You', or any nicknames.
				- The proposed allocation should include all {len(self.items)} items.
				- Do not include any extra text, only return the dictionary.

				If no proposal is found, return an empty dictionary.
				"""

				self.moderatorAgent.addToMemoryBuffer('user', inputText)
				
				try:
					response = self.moderatorAgent.run('user', self.moderatorAgent.systemInstructions)
					proposal_dict = ast.literal_eval(response)
					if isinstance(proposal_dict, dict):
						# Ensure that only valid agent names are used
						valid_agent_names = {self.name, other_agent_name}
						corrected_proposal = {}
						for item, agent_name in proposal_dict.items():
							if agent_name in valid_agent_names:
								corrected_proposal[item] = agent_name
							else:
								print(f"Invalid agent name '{agent_name}' for item '{item}'. Ignoring this item.")
						
						if corrected_proposal:
							return corrected_proposal
						
				except (ValueError, SyntaxError) as e:
					print(f"Failed to parse proposal on attempt {attempt + 1}: {str(e)}")
					if attempt < max_attempts - 1:
						print("Retrying...")
					continue
			
			print(f"Failed to parse proposal after {max_attempts} attempts. Returning empty dictionary.")
			return {}

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
					print(f"Removed {item.name} from {other_agent}")  # Debug statement
				self.board[agent_name].append(item)
				print(f"Added {item.name} to {agent_name}")  # Debug statement
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
		agent1FilePath = "CompetitiveAllocators/SystemInstructions.txt"
		agent2FilePath = "CompetitiveAllocators/BoulwareSystemInstructions.txt"
		self.items: list[Item] = items
		self.agent1 = Agent("Agent 1", agent1Model, agent1FilePath, useOpenAI=agent1UseOpenAI)
		self.agent2 = BoulwareAgent("Agent 2", agent2Model, agent2FilePath, items, self.agent1.name, useOpenAI=agent2UseOpenAI)
		self.moderatorAgent = Agent("Moderator", moderatorModel, "", useOpenAI=moderatorUseOpenAI)
		self.numItems = len(items)
		self.numConversationIterations = 0
		self.boardState = BoardState(self.agent1, self.agent2, items)
		self.consensusCounter = 0
  
		for item in self.items:
			# Remove agent names from preference statements to avoid confusion
			self.agent1.systemInstructions += f"\n- {item.name}: Your preference value for this item is {item.pref1} out of 1.0"
			self.agent2.systemInstructions += f"\n- {item.name}: Your preference value for this item is {item.pref2} out of 1.0"
		
		self.agent1.systemInstructions += "\n\nLet's begin! Remember to be concise."
		self.agent2.systemInstructions += "\n\nLet's begin! Remember to be concise."
  
		self.agent1.systemInstructions = self.agent1.systemInstructions.replace("[Your Name]", self.agent1.name).replace("[Opponent's Name]", self.agent2.name)
		self.agent2.systemInstructions = self.agent2.systemInstructions.replace("[Your Name]", self.agent2.name).replace("[Opponent's Name]", self.agent1.name)
  
		for item in self.items:
			# Add explicit instructions about item assignment format
			self.agent1.systemInstructions += (
				f"\n- {item.name}: Your preference value for this item is {item.pref1} out of 1.0. "
				f"The higher this value, the more you want this item for yourself."
			)
			self.agent2.systemInstructions += (
				f"\n- {item.name}: Your preference value for this item is {item.pref2} out of 1.0. "
				f"The higher this value, the more you want this item for yourself."
			)
		
		# Add clear format instructions for proposals
		proposal_format = """
		\nWhen proposing allocations:
		1. ALWAYS write your own name followed by the items you want for yourself
		2. THEN write your opponent's name followed by their items
		3. PRIORITIZE items with higher preference values for yourself
		4. Format example: 
		   Agent 1: Item X, Item Y
		   Agent 2: Item Z, Item W
		Where you replace items with those you actually want to allocate.
		"""
		
		self.agent1.systemInstructions += proposal_format
		self.agent2.systemInstructions += proposal_format
		
		self.agent1.systemInstructions += "\n\nLet's begin! Remember to be concise."
		self.agent2.systemInstructions += "\n\nLet's begin! Remember to be concise."

	def getItemIndex(self, item_name: str):
		for i, item in enumerate(self.items):
			if item.name.lower().strip() == item_name.lower().strip():
				return i
		return None

	def startNegotiation(self, maxIterations, startingAgent):
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
		iteration = 0

		deal_counter = 0  # Counts consecutive "Deal!" responses

		while not consensusReached and iteration < maxIterations:
			response = currentAgent.run("user", currentInput)
			if currentAgent == self.agent1:
				print(f"{Fore.LIGHTBLUE_EX}\n{currentAgent.name}: \n	{response}{Fore.RESET}")
			else:
				print(f"{Fore.LIGHTMAGENTA_EX}\n{currentAgent.name}: \n	{response}{Fore.RESET}")

			self.numConversationIterations += 1
			iteration += 1

			# Check if "Deal!" is in the response
			if "deal!" in response.lower():
				deal_counter += 1
				if deal_counter >= 2:
					print("\nBoth agents agreed. Extracting final allocation...")
					self.extractFinalAllocation()
					# After extraction, verify completeness
					agent1items = self.boardState.getItems(self.agent1.name)
					agent2items = self.boardState.getItems(self.agent2.name)
					allocated_items = {item.name for item in agent1items}.union({item.name for item in agent2items})
					all_items = {item.name for item in self.items}
					missing_items = all_items - allocated_items
					
					if missing_items:
						missing_str = ', '.join(missing_items)
						print(f"{Fore.RED}Missing Items Detected: {missing_str}. Continuing negotiation to allocate all items.{Fore.RESET}")
						currentInput = f"The following items were not allocated: {missing_str}. Please include them in your next proposal."
						deal_counter = 0  # Reset deal counter to continue negotiation
					else:
						consensusReached = True
						break
				else:
					currentInput = "Deal!"
			else:
				# Reset Deal! counter if response doesn't contain "Deal!"
				deal_counter = 0
				last_agent_said_deal = None

			# Swap agents for the next turn
			currentAgent = self.agent2 if currentAgent == self.agent1 else self.agent1
			currentInput = response

			if iteration == maxIterations:
				print("\nMaximum iterations reached without agreement.")

		if not consensusReached:
			print("\nNo agreement reached within the iteration limit.")

	def extractFinalAllocation(self):
		# Initialize board state
		self.boardState.resetItems()
		# Combine agents' memory
		conversation = self.agent1.memory + self.agent2.memory

		# Traverse the conversation to find the last proposed allocation
		allocation = {}
		for message in reversed(conversation):
			if isinstance(message, AIMessage) or isinstance(message, HumanMessage):
				content = message.content
				allocation = self.parseAllocationFromMessage(content)
				if allocation:
					break

		if allocation:
			# Assign items based on extracted allocation
			for item_name, agent_name in allocation.items():
				item_idx = self.getItemIndex(item_name)
				if item_idx is not None:
					self.boardState.addItem(self.items[item_idx], agent_name)
		else:
			print("Failed to extract allocation from conversation.")
			# Ensure that the fallback allocation is handled correctly
			# Possibly assign no items or handle as per requirements

	def parseAllocationFromMessage(self, message_content):
		# Parse the allocation from the message content
		allocation = {}
		lines = message_content.strip().split('\n')
		for line in lines:
			line = line.strip()
			if any(line.startswith(prefix) for prefix in ['Agent', 'You:', 'Me:']):
				# Split allocations separated by '|'
				parts = line.split('|')
				for part in parts:
					part = part.strip()
					sub_parts = part.split(':')	
					if len(sub_parts) == 2:
						prefix = sub_parts[0].strip()
						# Map prefixes to agent names
						if prefix == 'You':
							agent_name = self.agent2.name  # Changed from self.name to self.agent2.name
						elif prefix == 'Me':
							agent_name = self.agent1.name  # Changed from self.name to self.agent1.name
						else:
							agent_name = prefix
						items = [item.strip().rstrip('.') for item in sub_parts[1].split(',')]  # Strip trailing periods
						for item in items:
							allocation[item] = agent_name
		return allocation if allocation else None

	def printItems(self):
		print(f"\n{Fore.YELLOW}{self.agent1.name}'s items: {self.boardState.getItems(self.agent1.name)}")
		print(f"{self.agent2.name}'s items: {self.boardState.getItems(self.agent2.name)}{Fore.RESET}")

if __name__ == '__main__':
	pass  # Add code to initialize and start domain negotiation if necessary