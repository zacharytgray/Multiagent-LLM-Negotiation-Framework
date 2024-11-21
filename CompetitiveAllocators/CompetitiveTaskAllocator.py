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
import matplotlib.pyplot as plt  # Import for plotting

load_dotenv("keys.env")
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
		print(f"Debug - Opponent index: {opponentIndex}, Current index: {self.currentProposalIndex}")  # Added debug print
		return opponentIndex <= self.currentProposalIndex and opponentIndex >= 0

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
		# Convert item names to sets for comparison
		group1 = {item for item, agent in proposal.items()}
		group2 = set()
		for item_name, agent in proposal.items():
			if agent == self.agent2Name:  # If assigned to Agent 2
				group2.add(item_name)
				if item_name in group1:  # Remove from group1 if it was there
					group1.remove(item_name)
		
		# Now compare with ranked allocations
		for tier_index, tier in enumerate(self.rankedAllocations):
			for alloc in tier:
				alloc_group1 = {item.name for item in alloc[0]}
				alloc_group2 = {item.name for item in alloc[1]}
				if alloc_group1 == group1 and alloc_group2 == group2:
					return tier_index
		print(f"Error: Proposal not found in rankedAllocations.")
		print(f"Looking for - Group1: {group1}, Group2: {group2}")
		return -1

class BoulwareAgent(Agent):
	def __init__(self, name, model_name, systemInstructionsFilename, items, agent1Name, useOpenAI=False):  # Added agent1Name parameter
		super().__init__(name, model_name, systemInstructionsFilename, useOpenAI)
		self.items = items
		self.strategy = BoulwareStrategy(items, agent1Name, self.name)  # Passed agent1Name and self.name to strategy
		self.opponentOffer = {}  # Opponent's last offer

	def run(self, role, inputText):
		opponentUtility = 0  # Initialize opponentUtility to a default value
		self.addToMemoryBuffer(role, inputText)
		# Parse the opponent's proposal
		
		opponentOffer = self.parse_proposal(inputText)
		if opponentOffer != {}:
			self.opponentOffer = opponentOffer
			# Calculate utility of Agent 1's offer for Agent 2
			opponentUtility = sum(
				item.pref2 for item in self.items 
				if item.name in opponentOffer and opponentOffer[item.name] == self.name
			)

		 # **Handle 'Deal!' immediately**
		if "Deal!" in inputText:
			self.addToMemoryBuffer('assistant', "Deal!")
			return "Deal!"

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
			# Accept the opponent's offer
			self.addToMemoryBuffer('assistant', "Deal!")
			return "Deal!"

		# Generate a response using the model
		prompt = ChatPromptTemplate.from_messages(self.memory)
		chain = prompt | self.model	
		response = chain.invoke({})
		response_content = response.content if isinstance(response, AIMessage) else response
		self.addToMemoryBuffer('assistant', response_content)

		# Print the current proposal index and opponent's offer index
		currentProposalIndex = self.strategy.currentProposalIndex
		opponentOfferIndex = self.strategy.findProposalIndex(self.opponentOffer)  # Updated reference
  
		return response_content.strip()

	def parse_proposal(self, inputText):
		other_agent_name = 'Agent 1' if self.name != 'Agent 1' else 'Agent 2'
		if "Deal!" in inputText:
			# Opponent has accepted the deal
			print(f"{other_agent_name} has accepted the deal.")
			return {}  # Return an empty proposal

		if not hasattr(self, 'moderatorAgent'):
			if self.useOpenAI:
				self.moderatorAgent = Agent("Moderator", self.model_name, "", useOpenAI=self.useOpenAI)
			else:
				self.moderatorAgent = Agent("Moderator", self.model_name, "", useOpenAI=self.useOpenAI)

		# Clear the moderator's memory
		self.moderatorAgent.memory = []
		
		self.moderatorAgent.systemInstructions = f"""
		You have just received a message from {other_agent_name} in an item allocation negotiation.
		All items MUST be allocated in every proposal.

		Available items that MUST be allocated: {', '.join(item.name for item in self.items)}

		Rules:
		- Extract the proposed allocation from the message.
		- ALL items must be allocated - partial allocations are not allowed
		- Return the proposed allocation in a dictionary format: {{'Item A':'AGENT NAME', 'Item B':'AGENT NAME', ...}}
		- 'AGENT NAME' must be either '{self.name}' or '{other_agent_name}'
		- Do not include any extra text, only return the dictionary.
		- If the proposal is incomplete or missing items, return an empty dictionary.

		If no complete proposal is found, return an empty dictionary.
		"""

		self.moderatorAgent.addToMemoryBuffer('user', inputText)
		
		try:
			response = self.moderatorAgent.run('user', self.moderatorAgent.systemInstructions)
			proposal_dict = ast.literal_eval(response)
			if isinstance(proposal_dict, dict):
				# Ensure that only valid agent names are used
				valid_agent_names = {self.name, other_agent_name}
				corrected_proposal = {}
				
				# First pass: collect all valid items and their agents
				for item, agent_name in proposal_dict.items():
					item = item.strip()  # Clean up item name
					if agent_name in valid_agent_names:
						corrected_proposal[item] = agent_name
				
				# Verify both agents have at least one item
				if not all(agent in set(corrected_proposal.values()) for agent in valid_agent_names):
					print(f"Invalid proposal: Both agents must receive at least one item.")
					return {}
				
				# Check for missing items
				all_item_names = {item.name for item in self.items}
				proposed_item_names = set(corrected_proposal.keys())
				missing_items = all_item_names - proposed_item_names
				
				if missing_items:
					missing_items_str = ", ".join(missing_items)
					print(f"Incomplete proposal: Missing items: {missing_items_str}")
					self.addToMemoryBuffer('system', 
						f"Your proposal is incomplete. You forgot to allocate these items: {missing_items_str}. "
						"Please propose a complete allocation that includes ALL items: {', '.join(all_item_names)}")
					return {}  # Return empty dict for incomplete proposals
				
				return corrected_proposal
				
		except (ValueError, SyntaxError) as e:
			print(f"Failed to parse proposal: {str(e)}")
		
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
		self.iteration_numbers = []
		self.agent1_utilities = []
		self.agent2_utilities = []
		self.proposing_agents = []
  
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
		
		# Add clear format instructions for proposals with stronger emphasis on completeness
		all_items_str = ", ".join(item.name for item in items)
		proposal_format = f"""
		\nWhen negotiating:
		1. You can explain your reasoning and discuss preferences
		2. Be professional but conversational in your responses
		3. When making a proposal, format it as follows:
		   - First, explain your reasoning (optional but encouraged)
		   - Then clearly state your proposal using this format:
		   Agent 1: [items for Agent 1]
		   Agent 2: [items for Agent 2]
		4. ALL items must be allocated in every proposal: {all_items_str}
		5. You can counter-propose and explain why you think your proposal is fair
		6. You can refer to item preferences when explaining your proposals
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

		# Initialize current proposal
		currentProposal = None

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
					# Append to all lists to maintain consistent lengths
					self.iteration_numbers.append(iteration)
					self.agent1_utilities.append(self.agent1_utilities[-1] if self.agent1_utilities else 0)
					self.agent2_utilities.append(self.agent2_utilities[-1] if self.agent2_utilities else 0)
			else:
				# Reset Deal! counter if response doesn't contain "Deal!"
				deal_counter = 0
				last_agent_said_deal = None

			# Swap agents for the next turn
			currentAgent = self.agent2 if currentAgent == self.agent1 else self.agent1
			currentInput = response

			 # **Parse the proposed allocation after each response**
			proposal = self.parseAllocationFromMessage(response)
			if proposal: # prints most recent proposal found in conversation
				print(f"Proposal: {proposal}")
				# print agent 1 and 2's utilities for the proposal
				print(f"Agent 1 Utility: {sum(item.pref1 for item in self.items if proposal.get(item.name) == self.agent1.name)}")
				print(f"Agent 2 Utility: {sum(item.pref2 for item in self.items if proposal.get(item.name) == self.agent2.name)}")
				# **Calculate utilities based on the current proposal**
				agent1Utility = sum(
					item.pref1 for item in self.items
					if proposal.get(item.name) == self.agent1.name
				)
				agent2Utility = sum(
					item.pref2 for item in self.items
					if proposal.get(item.name) == self.agent2.name
				)

				 # Store which agent made this proposal
				self.proposing_agents.append("A1Prop" if currentAgent == self.agent1 else "A2Prop")
				self.iteration_numbers.append(iteration)
				self.agent1_utilities.append(agent1Utility)
				self.agent2_utilities.append(agent2Utility)
			else:
				# Append to all lists to maintain consistent lengths
				self.iteration_numbers.append(iteration)
				self.agent1_utilities.append(self.agent1_utilities[-1] if self.agent1_utilities else 0)
				self.agent2_utilities.append(self.agent2_utilities[-1] if self.agent2_utilities else 0)

		if not consensusReached:
			print("\nNo agreement reached within the iteration limit.")

	def generate_utility_plot(self):
		import matplotlib.pyplot as plt
		plt.figure()
		plt.plot(self.iteration_numbers, self.agent1_utilities, marker='o', color='blue', label='Agent 1 Utility')
		plt.plot(self.iteration_numbers, self.agent2_utilities, marker='o', color='red', label='Agent 2 Utility')
		plt.xlabel('Iteration Number')
		plt.ylabel('Agent Utility')
		plt.title('Agent Utilities per Iteration')
		plt.legend()
		plt.grid(True)
		plt.show()

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
		
		# First, find lines that contain agent allocations
		agent_lines = []
		for line in lines:
			line = line.strip()
			if any(line.lower().startswith(prefix.lower()) for prefix in ['Agent', 'You:', 'Me:']):
				agent_lines.append(line)
		
		# Process each agent line
		for line in agent_lines:
			# Split on ':' only for the first occurrence to handle item names that might contain colons
			parts = line.split(':', 1)
			if len(parts) == 2:
				agent_part = parts[0].strip()
				items_part = parts[1].strip()
				
				# Determine the agent name
				if agent_part.lower() == 'you':
					agent_name = self.agent2.name
				elif agent_part.lower() == 'me':
					agent_name = self.agent1.name
				else:
					agent_name = agent_part
				
				# Split items by comma and clean up each item name
				items = [item.strip().rstrip('.') for item in items_part.split(',') if item.strip()]
				
				# Add each item to the allocation dictionary
				for item in items:
					if item:  # Only add non-empty items
						allocation[item] = agent_name
		
		# Verify that we have a valid allocation (contains items for both agents)
		if allocation and all(agent in {self.agent1.name, self.agent2.name} for agent in set(allocation.values())):
			return allocation
		return None

	def printItems(self):
		print(f"\n{Fore.YELLOW}{self.agent1.name}'s items: {self.boardState.getItems(self.agent1.name)}")
		print(f"{self.agent2.name}'s items: {self.boardState.getItems(self.agent2.name)}{Fore.RESET}")

if __name__ == '__main__':
	pass  # Add code to initialize and start domain negotiation if necessary
