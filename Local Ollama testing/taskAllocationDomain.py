import ollama
from colorama import Fore

SYSTEM_INSTRUCTIONS = "You are about to be connected to a partner. You will collaborate with that partner to allocate tasks as efficiently as possible. Skill levels are on a scale from 1 to 10. Higher skill levels indicate greater proficiency. Initially, you know nothing about your partner's skill level. Conversationally determine the most suitable person for the task based only on skill level. Collaboration on a task is forbidden, and only one of you can be assigned a task at a time. Share your given skill levels directly when asked. Be clear and concise."

class Agent:
	def __init__(self, name) -> None:
		self.name = name
		self.temperature = 0.8
		self.tasks = []
		self.model = 'phi3:medium'
		self.memoryBuffer = []

	def addToMemoryBuffer(self, role, inputText): #role is either 'user', 'assistant', or 'system'
		self.memoryBuffer.append({'role':role, 'content': inputText})

	def queryModel(self):
		response = ollama.chat(model=self.model, messages=self.memoryBuffer, options = {'temperature': self.temperature,})
		return response['message']['content'].strip()
	
	def run(self, role, inputText):
		self.addToMemoryBuffer(role, inputText)
		response = self.queryModel()
		self.addToMemoryBuffer('assistant', response)
		if not response:
			print(f"{Fore.RED}Error: No response from model. Response: '{response}' {Fore.RESET}")
		return response
	
	def printMemoryBuffer(self, otherAgent):
		print(f"\n{Fore.YELLOW}{self.name}'s Memory Buffer:{Fore.RESET}")
		for dict in self.memoryBuffer:
			if dict.get('role') == 'system':
				print(Fore.RED + "System: " + dict.get('content') + Fore.RESET)
			elif dict.get('role') == 'user':
				print(Fore.BLUE + f"{otherAgent.name} (User): " + dict.get('content') + Fore.RESET)
			elif dict.get('role') == 'assistant':
				print(Fore.GREEN + f"{self.name} (Assistant): " + dict.get('content') + Fore.RESET)

	def addTask(self, task):
		self.tasks.append(task)
	
class Domain:
	def __init__(self, agent1, agent2) -> None:
		self.agent1 = agent1
		self.agent2 = agent2

	def printTasks(self):
		print(f"\n{Fore.YELLOW}{self.agent1.name}'s tasks: {self.agent1.tasks}")
		print(f"{self.agent2.name}'s tasks: {self.agent2.tasks}{Fore.RESET}")

	def getConsensus(self, task): #returns True if both agents agree on who should be assigned the task
		rawConsensus1 = self.agent1.run("system", f"Now, based on the conversation you just had with {self.agent2.name}, you must determine who is better suited for the task ({task}) based only on skill levels. Return 0 for {self.agent1.name} (Yourself) or 1 for {self.agent2.name} (Your partner). Do not say anything besides the number 0 or the number 1. Do not provide any explanation for your decision, only provide the appropriate number.").strip()
		rawConsensus2 = self.agent2.run("system", f"Now, based on the conversation you just had with {self.agent1.name}, you must determine who is better suited for the task ({task}) based only on skill levels. Return 0 for {self.agent1.name} (Your partner) or 1 for {self.agent2.name} (Yourself). Do not say anything besides the number 0 or the number 1. Do not provide any explanation for your decision, only provide the appropriate number.").strip()

		# Parse Agent 1 Response:
		if '1' in rawConsensus1: #if agent 1 thinks agent 2 should be assigned the task
			consensus1 = 1
		elif '0' in rawConsensus1: #if agent 1 thinks they should be assigned the task
			consensus1 = 0
		else: #if agent 1's response is invalid 
			consensus1 = f"Invalid input from {self.agent1.name} ({rawConsensus1}). Task not assigned."

		# Parse Agent 2 Response:
		if '1' in rawConsensus2: #if agent 2 thinks they should be assigned the task
			consensus2 = 1
		elif '0' in rawConsensus2: #if agent 2 thinks agent 1 should be assigned the task
			consensus2 = 0
		else: #if agent 2's response is invalid
			consensus2 = f"Invalid input from {self.agent2.name} ({rawConsensus2}). Task not assigned."

		# Consensus Logic:
		if consensus1 == 0 and consensus2 == 0: #if the agents agree that agent 1 should be assigned the task
			consensus = f"{self.agent1.name} has been assigned the task."
			self.agent1.addTask(task)
		elif consensus1 == 1 and consensus2 == 1: #if the agents agree that agent 2 should be assigned the task
			consensus = f"{self.agent2.name} has been assigned the task."
			self.agent2.addTask(task)
		else: #if the agents disagree
			consensus = f"Disagreement on who should be assigned the task. Task not yet assigned."
			print(f"\n{Fore.RED}Input: {consensus1}, {consensus2}{Fore.RESET}")

		print(f"\n{Fore.YELLOW}{consensus}{Fore.RESET}")

		return (consensus1 == consensus2), consensus1, consensus2
	
	def interruptConversation(self): #interrupt the conversation to allow the user to talk to agents directly
		#Type 1 to talk to agent 1
		#Type 2 to talk to agent 2
		#Type c to continue conversation
		#Type mb to see memory buffer
		userInput = ""
		while userInput.lower() != "c":
			userInput = input(f"\n{Fore.GREEN}What would you like to do? (1, 2, c, mb): {Fore.RESET}")
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

	def assignTask(self, task, skill1, skill2):
		print("\n" + "="*25 + f" NEW TASK: {task} " + "="*25)

		self.agent1.addToMemoryBuffer('system', f"NEW TASK: Your name is {self.agent1.name}. {SYSTEM_INSTRUCTIONS}. your skill level for this task, {task}, is {skill1} out of 10")
		self.agent2.addToMemoryBuffer('system', f"NEW TASK: Your name is {self.agent2.name}. {SYSTEM_INSTRUCTIONS}. your skill level for this task, {task}, is {skill2} out of 10")
		
		currentInput = f"Hello, I am {self.agent2.name}. Let's begin allocating our next task, {task}. "

		self.agent2.addToMemoryBuffer('assistant', currentInput)
	
		currentAgent = self.agent1

		consensusReached = False
		while not consensusReached:
			for i in range(3):
				response = currentAgent.run("user", currentInput)
				print(f"\n{currentAgent.name}'s response: \n	{response.strip()}")
				currentAgent = self.agent2 if currentAgent == self.agent1 else self.agent1
				currentInput = response

				if i == 2: # Manually add final dialogue to agent 2
					self.agent2.addToMemoryBuffer('user', currentInput)

				# self.interruptConversation() # uncomment to allow user to talk to agents directly inbetween messages or see memory buffers live
			
			consensus = self.getConsensus(task)
			if consensus[0]: # If agents agree on who should be assigned the task, end loop
				consensusReached = True
			else: # Make agents forget about consensus and encourage them to continue discussion
				# self.agent1.run("system", f"You both disagreed on who should get the task. You voted {consensus[1]}, and {self.agent2.name} voted {consensus[2]}. Continue discussion over the task ({task}) and consider reevaluating your decision based on skill level.")
				# self.agent2.run("system", f"You both disagreed on who should get the task. {self.agent1.name} voted {consensus[1]}, and you voted {consensus[2]}. Continue discussion over the task ({task}) and consider reevaluating your decision based on skill level.")
				self.agent1.addToMemoryBuffer('system', f"You both disagreed on who should get the task. You voted {consensus[1]}, and {self.agent2.name} voted {consensus[2]}. Continue discussion over the task ({task}) and consider reevaluating your decision based on skill level.")
				self.agent2.addToMemoryBuffer('system', f"You both disagreed on who should get the task. {self.agent1.name} voted {consensus[1]}, and you voted {consensus[2]}. Continue discussion over the task ({task}) and consider reevaluating your decision based on skill level.")

		
def main():
	agent1 = Agent("Igor")
	agent2 = Agent("Aslaug")
	domain = Domain(agent1, agent2)

	tasks = [("a word search", 6, 4), ("a math game", 4, 6), ("a card game", 9, 7)] # (task, skill1, skill2)
	for (task, skill1, skill2) in tasks:
		domain.assignTask(task, skill1, skill2)

	domain.printTasks()
	# agent1.printMemoryBuffer(otherAgent = agent2)
	# agent2.printMemoryBuffer(otherAgent = agent1)

if __name__ == main():
	main()