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

def parse_allocation(message_content, agent1_name, agent2_name, items):
	message_content = message_content.replace('*', '')  # Remove asterisks
	lines = message_content.strip().split('\n')
 
	allocation = {}
	
	# Find the "PROPOSAL:" marker
	proposal_index = -1
	for i, line in enumerate(lines):
		if line.strip().upper() == "PROPOSAL:":
			proposal_index = i
			break

	hasDeal = "deal!" in message_content.lower()

	if proposal_index != -1:
		proposal_lines = lines[proposal_index + 1], lines[proposal_index + 2]
		agent_lines = []
		for line in proposal_lines:
			line = line.strip()
			if line.lower().startswith('agent'): # Process all lines that start with "Agent"
				agent_lines.append(line)
		
		# Process agent lines outside the loop
		for line in agent_lines:
			parts = line.split(':', 1)
			
			if len(parts) != 2:
				print("Invalid proposal format: Missing colon")
				print("Parsed Parts:", parts)
				return None, hasDeal

			agent_part = parts[0].strip().lower()
			items_part = parts[1].strip()
	
			if agent_part == agent1_name.lower():
				agent_name = agent1_name
			elif agent_part == agent2_name.lower():
				agent_name = agent2_name
			else:
				print(f"Invalid agent name: {agent_part}")
				print("Parsed Agent Part:", agent_part)
				return None, hasDeal
	
			items_list = [item.strip() for item in items_part.split(',') if item.strip()]
			for item in items_list:
				allocation[item] = agent_name

		if len(allocation) != len(items):
			print("Invalid proposal: Must include all items")
			print("Parsed Allocation:", allocation)
			return None, hasDeal

		if len(agent_lines) != 2: # Ensure there are exactly two agent lines
			print("Invalid proposal format: Must have exactly two agent lines")
			print("Parsed Agent Lines:", agent_lines)
			return None, hasDeal
		
	return (allocation if allocation else None), hasDeal

class Agent:
	def __init__(self, name, model_name, systemInstructionsFilename, useOpenAI=False):
		self.name = name
		self.model_name = model_name
		self.numTokensGenerated = 0
		self.memory = []  # Replace ConversationBufferMemory with a simple list
		self.useOpenAI = useOpenAI
		self.systemInstructions = f"Your name is {self.name}.\n"
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
				# self.addToMemoryBuffer('system', self.systemInstructions)
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
		self.agent2Name = agent2Name  
		self.currentProposal = None  # Agent 2's current planned proposal
		self.currentUtility = 0  # Utility of currentProposal
		self.currentProposalIndex = 0  # Index in utility rankings
		self.opponentOffer = None  # Opponent's last proposal
		self.opponentProposalIndex = None  # Index in utility rankings
		self.rankedAllocations = self.getAllocationRankings() # Ranked allocations by utility (0th index is highest utility for Agent 2)
		self.numIterations = 0  # Number of rounds passed

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

	def shouldAccept(self):
		if self.opponentProposalIndex is None:
			print("Error: Opponent proposal index not set.")
			return False
		if self.opponentProposalIndex <= self.nextProposalIndex(self.numIterations + 1):
			if self.opponentProposalIndex >= 0:
				return True	
			else:
				print(f"Error: Opponent proposal index is invalid: {self.opponentProposalIndex}")
				return False
		else:
			return False

	def nextProposalIndex(self, numIterations):
		totalTiers = len(self.rankedAllocations)
		maxTierIndex = totalTiers - 1
		if numIterations == 1:
			nextTierIndex = 0
		else:
			nextTierIndex = min(
				int(maxTierIndex * (1 - (0.9 ** (numIterations - 1)))), 
				maxTierIndex
			)
		return nextTierIndex

	def updateCurrentAllocation(self, numIterations):
		self.numIterations = numIterations
		totalTiers = len(self.rankedAllocations)
		maxTierIndex = totalTiers - 1
		# Ensure that for the first round, currentTierIndex is 0
		if numIterations == 1:
			currentTierIndex = 0
		else:
			currentTierIndex = min(
				int(maxTierIndex * (1 - (0.9 ** (numIterations - 1)))), 
				maxTierIndex
			)
		self.currentProposalIndex = currentTierIndex
		self.currentProposal = self.rankedAllocations[currentTierIndex][0]  # Always select the top allocation in the tier

	def findProposalIndex(self, proposal):
		# Correctly process the proposal to separate items for Agent 1 and Agent 2
		group1 = set()
		group2 = set()
		agent1_name_clean = self.agent1Name.strip().lower()
		agent2_name_clean = self.agent2Name.strip().lower()
		for item_name, agent in proposal.items():
			agent_clean = agent.strip().lower()
			if agent_clean == agent1_name_clean:
				group1.add(item_name.strip())
			elif agent_clean == agent2_name_clean:
				group2.add(item_name.strip())
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

def printMemoryBuffer(agent):
	"""Print an agent's memory buffer with color-coded messages"""
	print(f"\n{Fore.CYAN}=== Memory Buffer for {agent.name} ===")
	for idx, msg in enumerate(agent.memory, 1):
		if isinstance(msg, SystemMessage):
			color = Fore.YELLOW  # System messages in yellow
			msg_type = "SYSTEM"
		elif isinstance(msg, HumanMessage):
			color = Fore.GREEN  # User messages in green
			msg_type = "USER"
		elif isinstance(msg, AIMessage):
			color = Fore.MAGENTA  # AI/Assistant messages in magenta
			msg_type = "ASSISTANT"
		else:
			color = Fore.WHITE  # Unknown message types in white
			msg_type = "UNKNOWN"
			
		print(f"\n{color}[Message {idx} - {msg_type}]")
		print(f"{msg.content}")
		print(f"{'-'*50}{Fore.RESET}")
	print(f"{Fore.CYAN}{'='*50}{Fore.RESET}\n")

class BoulwareAgent(Agent):
	def __init__(self, name, model_name, systemInstructionsFilename, items, agent1Name, useOpenAI=False):  # Added agent1Name parameter
		super().__init__(name, model_name, systemInstructionsFilename, useOpenAI)
		self.items = items
		self.strategy = BoulwareStrategy(items, agent1Name, self.name)  # Passed agent1Name and self.name to strategy
		self.opponentOffer = {}  # Opponent's last offer

	def run(self, role, inputText):
		self.memory = [msg for msg in self.memory if isinstance(msg, SystemMessage)]
		self.addToMemoryBuffer(role, inputText)
		other_agent_name = 'Agent 1' if self.name != 'Agent 1' else 'Agent 2'
		newOpponentOffer, hasDeal = parse_allocation(inputText, other_agent_name, self.name, self.items)
	
		if hasDeal:
			# Convert currentProposal tuple to dictionary format
			current_proposal_dict = {}
			# Add Agent 1's items
			for item in self.strategy.currentProposal[0]:
				current_proposal_dict[item.name] = other_agent_name
			# Add Agent 2's items
			for item in self.strategy.currentProposal[1]:
				current_proposal_dict[item.name] = self.name
				
			self.opponentOffer = current_proposal_dict
			self.strategy.opponentOffer = self.opponentOffer
			self.strategy.opponentProposalIndex = self.strategy.currentProposalIndex
		elif newOpponentOffer:
			self.opponentOffer = newOpponentOffer
			self.strategy.opponentProposalIndex = self.strategy.findProposalIndex(self.opponentOffer)
			self.strategy.opponentOffer = self.opponentOffer

		if self.strategy.shouldAccept():
			print(f"{Fore.GREEN}Should Accept Deal{Fore.RESET}")
			group1_items = [item_name for item_name, agent in self.opponentOffer.items() 
						  if agent == other_agent_name]
			group2_items = [item_name for item_name, agent in self.opponentOffer.items() 
						  if agent == self.name]
			systemMessage = (
				f"{Fore.YELLOW}IMPORTANT: You must say 'Deal!' and restate this exact allocation:\n"
				f"PROPOSAL:\n"
				f"{other_agent_name}: {', '.join(group1_items)}\n"
				f"{self.name}: {', '.join(group2_items)}"
			)
		else:	
			print(f"{Fore.RED}Should Decline Deal{Fore.RESET}")
			self.strategy.updateCurrentAllocation(self.strategy.numIterations + 1)
			group1_items = [item.name for item in self.strategy.currentProposal[0]]
			group2_items = [item.name for item in self.strategy.currentProposal[1]]
			
			# Strengthen the system message to be more explicit
			systemMessage = (
				f"{Fore.YELLOW}CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS EXACTLY:\n"
				f"1. You must decline the current proposal\n"
				f"2. You must then copy and paste this EXACT allocation without any changes:\n"
				f"PROPOSAL:\n"
				f"{other_agent_name}: {', '.join(group1_items)}\n"
				f"{self.name}: {', '.join(group2_items)}\n"
				f"3. Do not modify the items or allocations in any way\n"
				f"4. Do not add or remove any items from the allocation"
			)
			
		# Add response validation and retry logic
		max_retries = 3
		for attempt in range(max_retries):
			self.addToMemoryBuffer('system', systemMessage)
			prompt = ChatPromptTemplate.from_messages(self.memory)
			chain = prompt | self.model	
			response = chain.invoke({})
			response_content = response.content if isinstance(response, AIMessage) else response
			
			# Validate the response
			allocation, hasDeal = parse_allocation(response_content, other_agent_name, self.name, self.items)
			
			if allocation:
				# Check if allocation matches required allocation
				required_group1 = set(group1_items)
				required_group2 = set(group2_items)
				actual_group1 = {item_name for item_name, agent in allocation.items() 
							   if agent == other_agent_name}
				actual_group2 = {item_name for item_name, agent in allocation.items() 
							   if agent == self.name}
				print(f"{Fore.YELLOW}Expected Allocation: \nAgent 1: {required_group1} \nAgent 2: {required_group2}")
				print(f"\nAgent 1's Offer Index: {self.strategy.opponentProposalIndex}")
				print(f"Agent 2's New Allocation Index: {self.strategy.currentProposalIndex}")
				if (required_group1 == actual_group1 and 
					required_group2 == actual_group2):
					# Valid response that matches requirements
					self.addToMemoryBuffer('assistant', response_content)
					return response_content.strip()
					
			if attempt < max_retries:
				# Try again with stronger message
				systemMessage = (
					f"{Fore.RED}WARNING: Your previous response did not match the required allocation.\n"
					f"You MUST use the EXACT allocation specified. No modifications allowed.\n"
					)
				if self.strategy.shouldAccept():
					systemMessage += (f"You should say 'Deal!' in your response to agree to the current proposal and restate the allocation.")
				systemMessage += (
					f"Required allocation:\n"
					f"PROPOSAL:\n"
					f"{other_agent_name}: {', '.join(group1_items)}\n"
					f"{self.name}: {', '.join(group2_items)}"
					)
				# Clear the failed response from memory
				self.memory.pop()
			else:
				# If we've exhausted retries, use the last response
				self.addToMemoryBuffer('assistant', response_content)
				return response_content.strip()

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
		self.iteration_numbers = [] # Store iteration numbers for plotting
		self.agent1_utilities = [] # Store Agent 1 utilities for plotting
		self.agent2_utilities = [] # Store Agent 2 utilities for plotting
		self.proposing_agents = [] # Store the agent who made each proposal for plotting
  
		for item in self.items:
			# Remove agent names from preference statements to avoid confusion
			self.agent1.systemInstructions += f"\n- {item.name}: Your preference value for this item is {item.pref1} out of 1.0"
			self.agent2.systemInstructions += f"\n- {item.name}: Your preference value for this item is {item.pref2} out of 1.0"
		
		self.agent1.systemInstructions += "\n\nLet's begin! Remember to be concise."
		self.agent2.systemInstructions += "\n\nLet's begin! Remember to be concise."
  
		self.agent1.systemInstructions = self.agent1.systemInstructions.replace("[Your Name]", self.agent1.name).replace("[Opponent's Name]", self.agent2.name)
		self.agent2.systemInstructions = self.agent2.systemInstructions.replace("[Your Name]", self.agent2.name).replace("[Opponent's Name]", self.agent1.name)
  
		# # Add clear format instructions for proposals with stronger emphasis on completeness
		# all_items_str = ", ".join(item.name for item in items)
		# proposal_format = f"""
		# \nWhen negotiating:	
		# 1. You can explain your reasoning and discuss preferences
		# 2. Be professional but conversational in your responses
		# 3. When making a proposal, format it as follows:
		#    - First, explain your reasoning (optional but encouraged)
		#    - Then clearly state your proposal using this format:
		# 	Proposal:
		# 	Agent 1: [items for Agent 1]
		# 	Agent 2: [items for Agent 2]
		# 4. ALL items must be allocated in every proposal: {all_items_str}
		# 5. You can counter-propose and explain why you think your proposal is fair
		# 6. You can refer to item preferences when explaining your proposals
		# """
		
		# self.agent1.systemInstructions += proposal_format
		# self.agent2.systemInstructions += proposal_format

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
		# currentAgent.addToMemoryBuffer('user', currentInput)
	
		consensusReached = False
		deal_counter = 0  # Counts consecutive "Deal!" responses

		while not consensusReached and self.numConversationIterations < maxIterations:
			response = currentAgent.run("user", currentInput)
			allocation, hasDeal = parse_allocation(response, self.agent1.name, self.agent2.name, self.items)

			if currentAgent == self.agent1:
				print(f"{Fore.LIGHTBLUE_EX}\n{currentAgent.name}: \n	{response}{Fore.RESET}")
			else:
				print(f"{Fore.LIGHTMAGENTA_EX}\n{currentAgent.name}: \n	{response}{Fore.RESET}")

			self.numConversationIterations += 1
   
			# Calculate and store utilities if there's a valid allocation
			if allocation:
				agent1_utility = sum(
					item.pref1 for item in self.items 
					if allocation.get(item.name) == self.agent1.name
				)
				agent2_utility = sum(
					item.pref2 for item in self.items
					if allocation.get(item.name) == self.agent2.name
				)
				
				self.iteration_numbers.append(self.numConversationIterations)
				self.agent1_utilities.append(agent1_utility)
				self.agent2_utilities.append(agent2_utility)
				self.proposing_agents.append(currentAgent.name)

				print(f"Agent 1 Utility: {agent1_utility}")
				print(f"Agent 2 Utility: {agent2_utility}")
   
			if hasDeal:
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
						currentInput = f"The following items were not allocated: {missing_str}. Please include them in your next proposal."
						deal_counter = 0  # Reset deal counter to continue negotiation
					else:
						consensusReached = True
						break
			else:
				# Reset Deal! counter if response doesn't contain "Deal!"
				deal_counter = 0

			# Swap agents for the next turn
			currentAgent = self.agent2 if currentAgent == self.agent1 else self.agent1
			currentInput = response
   
		if not consensusReached:
			print("\nNo agreement reached within the iteration limit.")
   
		printMemoryBuffer(self.agent1)

	def generate_utility_plot(self):
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots()
		
		# Plot utility lines
		ax.plot(self.iteration_numbers, self.agent1_utilities, marker='o', color='blue', label='Agent 1 Utility')
		ax.plot(self.iteration_numbers, self.agent2_utilities, marker='o', color='red', label='Agent 2 Utility')
		
		# For each point, add an 'X' marker if it was the starting agent
		for i, (iter_num, agent1_util, agent2_util) in enumerate(zip(self.iteration_numbers, self.agent1_utilities, self.agent2_utilities)):
			proposer = self.proposing_agents[i]
			if proposer == "Agent 1":
				ax.annotate('X', (iter_num, agent1_util), xytext=(0, 10),
						   textcoords='offset points', ha='center')
			else:
				ax.annotate('X', (iter_num, agent2_util), xytext=(0, -10),
						   textcoords='offset points', ha='center')

		ax.set_xlabel('Iteration Number')
		ax.set_ylabel('Agent Utility')
		ax.set_title('Agent Utilities per Iteration')
		ax.legend()
		ax.grid(True)
		plt.show()

	def extractFinalAllocation(self):
		# Initialize board state
		self.boardState.resetItems()
		
		# Combine agents' memory
		conversation = self.agent1.memory + self.agent2.memory
		
		# Traverse the conversation backwards to find the last valid allocation
		last_allocation = None
		for message in reversed(conversation):
			if isinstance(message, (AIMessage, HumanMessage)):
				content = message.content
				parsed, hasDeal = parse_allocation(content, self.agent1.name, self.agent2.name, self.items)
				if isinstance(parsed, dict):  # Only store if it's a valid allocation dictionary
					if parsed:
						last_allocation = parsed
						break  # Stop at the first valid allocation we find
		
		if last_allocation:
			# Assign items based on the last valid allocation
			for item_name, agent_name in last_allocation.items():
				item_idx = self.getItemIndex(item_name)
				if item_idx is not None:
					self.boardState.addItem(self.items[item_idx], agent_name)
			print("\nFinal allocation extracted successfully.")
		else:
			print("\nFailed to extract allocation from conversation.")

	def printItems(self):
		print(f"\n{Fore.YELLOW}{self.agent1.name}'s items: {self.boardState.getItems(self.agent1.name)}")
		print(f"{self.agent2.name}'s items: {self.boardState.getItems(self.agent2.name)}{Fore.RESET}")

if __name__ == '__main__':
	pass  # Add code to initialize and start domain negotiation if necessary
