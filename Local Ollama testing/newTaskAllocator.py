import ollama
from colorama import Fore

SYSTEM_INSTRUCTIONS = "You are about to be connected to a partner. You will collaborate with that partner to allocate tasks as efficiently as possible solely based on both of your skill levels. Skill levels are on a scale from 1 to 10. Higher skill levels indicate greater proficiency. Initially, you know nothing about your partner's skill level. Conversationally determine the most suitable person for the task based on skill level. You are not allowed to work together on a task. Be clear and concise. Share your given skill levels directly when asked, and do not discuss your abilities."
# SYSTEM_INSTRUCTIONS = "You are about to be connected to a partner. You will collaborate with that partner to allocate tasks as efficiently as possible. Skill levels are on a scale from 1 to 10. Higher skill levels indicate greater proficiency. Initially, you know nothing about your partner's skill level. Conversationally determine the most suitable person for the task based only on skill level. Collaboration on a task is forbidden. Share your given skill levels directly when asked. Be clear and concise."

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

		if rawConsensus1 == rawConsensus2:
			if rawConsensus1 == '0':
				consensus = f"{self.agent1.name} has been assigned the task."
				self.agent1.addTask(task)
			elif rawConsensus1 == '1':
				consensus = f"{self.agent2.name} has been assigned the task"
				self.agent2.addTask(task)
			else:
				consensus = f"Invalid input ({rawConsensus1}). Task not assigned."
		else:
			consensus = f"Disagreement on who should be assigned the task. Task not yet assigned."
			print(f"{Fore.RED}Input: {rawConsensus1}, {rawConsensus2}{Fore.RESET}")

		print(f"\n{Fore.YELLOW}{consensus}{Fore.RESET}")

		return rawConsensus1 == rawConsensus2
	
	def interruptConversation(self): #interrupt the conversation to allow the user to talk to agents directly
		userInput = ""
		while userInput.lower() != "q":
			userInput = input("Which agent would you like to talk to? (1, 2 or Q): ")
			if userInput != "q":
				if userInput == "1":
					userInput = input("What would you like to say to agent 1?: ")
					response = self.agent1.run("user", userInput)
					print(f"{self.agent1.name}: {response}")
				elif userInput == "2":
					userInput = input("What would you like to say to agent 2?: q")
					response = self.agent2.run("user", userInput)
					print(f"{self.agent1.name}: {response}")
				else:
					print("Pass...")

	def assignTask(self, task, skill1, skill2):
		print("\n" + "="*25 + f" NEW TASK: {task} " + "="*25)

		self.agent1.addToMemoryBuffer('system', f"Your name is {self.agent1.name}. {SYSTEM_INSTRUCTIONS}. your skill level for this task, {task}, is {skill1}")
		self.agent2.addToMemoryBuffer('system', f"Your name is {self.agent2.name}. {SYSTEM_INSTRUCTIONS}. your skill level for this task, {task}, is {skill2}")
		
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

				# self.interruptConversation() # uncomment to allow user to talk to agents directly inbetween messages
			
			if self.getConsensus(task): # If agents agree on who should be assigned the task, end loop
				consensusReached = True
			else: # Make agents forget about consensus and encourage them to continue discussion
				self.agent1.memoryBuffer = self.agent1.memoryBuffer[:-2]
				self.agent2.memoryBuffer = self.agent2.memoryBuffer[:-2]
				self.agent1.run("system", "You both disagreed on who should get the task. Reevaluate your decision based on skill level.")
				self.agent2.run("system", "You both disagreed on who should get the task. Reevaluate your decision based on skill level.")
		
def main():
	agent1 = Agent("Igor")
	agent2 = Agent("Aslaug")
	domain = Domain(agent1, agent2)

	domain.assignTask(task = "a word search", skill1 = 6, skill2 = 4) #agent1 should be assigned
	# domain.assignTask(task = "a math game", skill1 = 4, skill2 = 6) #agent2 should be assigned
	# domain.assignTask(task = "a card game", skill1 = 9, skill2 = 7) #agent1 should be assigned

	domain.printTasks() #agent1 should have "a word search" and "a card game", agent2 should have "a math game"

	agent1.printMemoryBuffer(otherAgent = agent2)
	agent2.printMemoryBuffer(otherAgent = agent1)

if __name__ == main():
	main()