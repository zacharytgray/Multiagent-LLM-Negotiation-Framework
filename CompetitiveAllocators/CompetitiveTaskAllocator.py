import ast
import ollama
from colorama import Fore
import concurrent.futures

def logAssignedItems(fileName, boardState):
    agent1Name = boardState.agent1.name
    agent2Name = boardState.agent2.name
    agent1Items = boardState.getItems(agent1Name)
    agent2Items = boardState.getItems(agent2Name)
    f = open(fileName, "a")
    f.write(f"\n{agent1Name}'s items: {agent1Items}\n")
    f.write(f"{agent2Name}'s items: {agent2Items}\n")
    f.close()

class Agent:
	def __init__(self, name, model, systemInstructionsFilename) -> None:
		self.name = name
		self.numTokensGenerated = 0
		self.memoryBuffer = []
		self.model = model
		self.temperature = 0.3
		self.instructionsFilename = systemInstructionsFilename
		self.systemInstructions = f"Your name is {self.name}. "

		if systemInstructionsFilename != "":
			try:
				with open(self.instructionsFilename, "r") as f:
					self.systemInstructions += f.read()
				self.addToMemoryBuffer('system', self.systemInstructions)
			except FileNotFoundError:
				print(f"Error: {self.instructionsFilename} not found.")
				exit(1)
		
	def addToMemoryBuffer(self, role, inputText): #role is either 'user', 'assistant', or 'system'
		self.memoryBuffer.append({'role':role, 'content': inputText})
  
	def validateMessages(self, messages):
		valid_roles = {"system", "user", "assistant"}
		for message in messages:
			if "role" not in message or message["role"] not in valid_roles:
				print(messages)
				raise ValueError(f"Invalid message role: {message}")

	def queryModel(self):
		def model_query():
			self.validateMessages(self.memoryBuffer)  # Validate messages
			response = ollama.chat(model=self.model, messages=self.memoryBuffer, options = {'temperature': self.temperature, 'num_predict': 100},)
			self.numTokensGenerated += response['eval_count']
			return response['message']['content'].strip()
		with concurrent.futures.ThreadPoolExecutor() as executor:
			future = executor.submit(model_query)
			try:
				numMinutesTimeout = 8 # minutes
				return future.result(timeout = (60 * numMinutesTimeout))
			except concurrent.futures.TimeoutError:
				print(f"{Fore.RED}Error: Timeout in model query.{Fore.RESET}")
				return "TIMEOUTERROR"
		
	def run(self, role, inputText):
		self.addToMemoryBuffer(role, inputText)
		timeoutStr = "You took too long to respond. Please try again, avoiding any infinite loops."
      			
		withinTimeLimit = False
		hasResponse = False
		while not withinTimeLimit or not hasResponse:
			withinTimeLimit = False
			hasResponse = False
			response = self.queryModel()
   
			if response:
				hasResponse = True
			else:
				print(f"{Fore.RED}Error: No response from {self.name}.{Fore.RESET}")
    
			if response == "TIMEOUTERROR":
				print(f"{Fore.RED}Timeout: {self.name} took too long to respond. Trying again.{Fore.RESET}")
				self.addToMemoryBuffer('user', timeoutStr)
			else:
				withinTimeLimit = True
				self.addToMemoryBuffer('assistant', response)
    
		return response.strip()
	
	def printMemoryBuffer(self):
		print(f"\n{Fore.YELLOW}{self.name}'s Memory Buffer:{Fore.RESET}")
		for dict in self.memoryBuffer:
			if dict.get('role') == 'system':
				print(Fore.RED + "System: " + dict.get('content') + Fore.RESET)
			elif dict.get('role') == 'user':
				print(Fore.LIGHTBLUE_EX + f"User: " + dict.get('content') + Fore.RESET)
			elif dict.get('role') == 'assistant':
				print(Fore.LIGHTGREEN_EX + f"{self.name} (Assistant): " + dict.get('content') + Fore.RESET)
			else:
				print(f"{Fore.RED}Error: Invalid role in memory buffer: {dict.get('role')}")
				print(f"Content: {dict.get('content')}{Fore.RESET}")

class Item:
    def __init__(self, name, pref1, pref2):
        self.name = name
        self.pref1 = pref1
        self.pref2 = pref2
    
    def __repr__(self):
        return f"{self.name} ({self.pref1}, {self.pref2})"
        
class BoardState:
	def __init__(self, agent1: Agent, agent2: Agent, items: list[Item]):
		self.agent1 = agent1
		self.agent2 = agent2
		self.items = [item for item in items]
  
		self.board = {
			f"{agent1.name}": [Item],
			"unassigned": items,
			f"{agent2.name}": [Item]
		}
    
	def getItems(self, row: str): # row is either agent1.name, agent2.name, or unassigned
		return self.board[row]

	def resetItems(self):
		self.board[self.agent1.name] = []
		self.board[self.agent2.name] = []
		self.board["unassigned"] = []
		for item in self.items:
			self.board["unassigned"].append(item)

	def addItem(self, item: Item, row):
		if item not in self.board[row]:
			if item in self.items and item not in self.board[row]:
				self.board[row].append(item)
				self.board["unassigned"].remove(item)
			else:
				print(f"Error: {item} not in item list.")
		else:
			print(f"Error: {item} already in {row} item list.")
   
		return item
		
	
class Domain:
	def __init__(self, items: list[Item], model: str) -> None:
		self.agent1 = Agent("Agent 1", model, "CompetitiveAllocators/Agent1CompetitiveSystemInstructions.txt")
		self.agent2 = Agent("Agent 2", model, "CompetitiveAllocators/Agent2CompetitiveSystemInstructions.txt")
		self.moderatorAgent = Agent("Moderator", model, "")
		self.items: list[Item] = items
		self.numItems = len(items)
		self.numConversationIterations = 0
		self.boardState = BoardState(self.agent1, self.agent2, items)
  
		for item in self.items:
			self.agent1.systemInstructions +=  f"\n- {item.name}: {self.agent1.name}, Your preference value for this item is {item.pref1} out of 1.0"
			self.agent2.systemInstructions +=  f"\n- {item.name}: {self.agent2.name}, Your preference value for this item is {item.pref2} out of 1.0"
		
		self.agent1.systemInstructions += "\n\nLet's begin! Remember to be concise."
		self.agent2.systemInstructions += "\n\nLet's begin! Remember to be concise."
  
	def getItemIndex(self, itemName):
		for i in range(len(self.items)):
			if self.items[i].name.lower().strip() == itemName:
				return i
		return

	def getConsensus(self, boardState):
		boardState.resetItems()
		self.moderatorAgent.memoryBuffer = []
		self.moderatorAgent.systemInstructions = f"""
You have just received a conversation between {self.agent1.name} and {self.agent2.name}. You are the moderator for this item allocation negotiation.
These two partners have been asked to allocate {self.numItems} items between each other based on their own preferences for each item.
Rules:
- You are not going to change their decisions in any way. 
- Your job is to simply show me the results of their conversation in a dictionary format: {{'Item A':'AGENT NAME', 'Item B':'AGENT NAME', ...}}
- Return the results that they seem to agree on the most. If they do not agree on an item, return 'TBD' for that item.
- If they decide to collaborate or split an item, return 'COLLAB' for that item.
- Note that 'AGENT NAME' is a placeholder for the name of the agent you think should be assigned that item based on their conversation. It should be replaced with '{self.agent1.name}', '{self.agent2.name}', or 'TBD' if they have not come to a consensus on that item.
- Include apostrophes as shown around the item names and agent names to ensure the dictionary is formatted correctly.
- To ensure your message is a valid dictionary, make sure your response starts with an open curly bracket and ends with a curly bracket. 
- Do not inculde leading or trailing apostrophes. 
- Do not inclue any headers like 'python' or 'json'.
- Do not respond with any extra text, not even an introduction. Simply return the following, replacing 'AGENT NAME' as needed:

"""

		self.moderatorAgent.systemInstructions += "{"
		for item in self.items:
			self.moderatorAgent.systemInstructions += f"'{item.name}':'AGENT NAME'," 
		self.moderatorAgent.systemInstructions = self.moderatorAgent.systemInstructions[:-1]
		self.moderatorAgent.systemInstructions += "}"
   
		memoryBuffer = self.agent1.memoryBuffer 
		for dialogue in memoryBuffer:
			if dialogue.get('role') == 'user':
				self.moderatorAgent.addToMemoryBuffer('user', f"{self.agent2.name}'s Response: " + dialogue.get('content'))
			elif dialogue.get('role') == 'assistant':
				self.moderatorAgent.addToMemoryBuffer('user', f"{self.agent1.name}'s Response: " + dialogue.get('content'))

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

		if len(boardState.getItems(self.agent1.name)) + len(boardState.getItems(self.agent2.name)) > self.numItems:
			print(f"{Fore.RED}Error: Too many items assigned.")
			print(f"{self.agent1.name}'s items: {boardState.getItems(self.agent1.name)}")
			print(f"{self.agent2.name}'s items: {boardState.getItems(self.agent2.name)}{Fore.RESET}")
			return False
		elif len(boardState.getItems(self.agent1.name)) + len(boardState.getItems(self.agent2.name)) < self.numItems:
			print(f"{Fore.RED}Error: Not all items assigned.")
			print(f"{self.agent1.name}'s items: {boardState.getItems(self.agent1.name)}")
			print(f"{self.agent2.name}'s items: {boardState.getItems(self.agent2.name)}{Fore.RESET}")
			return False
  
		return True

	def printItems(self):
		print(f"\n{Fore.YELLOW}{self.agent1.name}'s items: {self.boardState.getItems(self.agent1.name)}")
		print(f"{self.agent2.name}'s items: {self.boardState.getItems(self.agent2.name)}{Fore.RESET}")
	

	def interruptConversation(self): #interrupt the conversation to allow the user to talk to agents directly
		#Type 1 to talk to agent 1
		#Type 2 to talk to agent 2
		#Type mb to see memory buffer
		#Type c to continue conversation

		userInput = ""
		while userInput.lower() != "c":
			userInput = input(f"\n{Fore.GREEN}What would you like to do? (1, 2, mb, c): {Fore.RESET}")
			if userInput != "c":
				if userInput == "1": #user wants to talk to agent 1
					userInput = input(f"Chat to Agent 1 ({self.agent1.name}): ")
					response = self.agent1.run("user", userInput)
					print(f"{self.agent1.name}: {response}")
				elif userInput == "2": #user wants to talk to agent 2
					userInput = input(f"Chat to Agent 2 ({self.agent2.name}): ")
					response = self.agent2.run("user", userInput)
					print(f"{self.agent2.name}: {response}")
				elif userInput == "mb": #user wants to see memory buffer
					self.agent1.printMemoryBuffer(self.agent2)
					self.agent2.printMemoryBuffer(self.agent1)
				else: 
					print("Pass...")

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

				# uncomment to allow user to talk to agents directly inbetween messages or see memory buffers live
				# self.interruptConversation() 
    
			print(f"\nAsking Moderator for Consensus...")
   
			consensusReached = self.getConsensus(self.boardState)
   
def main():
	numIterations = 3
	agent1 = Agent("Agent 1")
	agent2 = Agent("Agent 2")
 
	# Pref levels are on a scale from 0.0 to 1.0, where 1.0 is the most prefered item and 0.0 is the least preferred item
	items = [('Item A', 3, 1), ('Item B', 2, 2), ('Item C', 1, 4), ('Item D', 4, 3)] # formatted as [('Task X', Pref1, Pref2), ...]
	items = [Item(item[0], item[1], item[2]) for item in items]
	domain = Domain(agent1, agent2, items)
	domain.startNegotiation(numIterations)

	# agent1.printMemoryBuffer(otherAgent = agent2)
	# agent2.printMemoryBuffer(otherAgent = agent1)
 
# main()