import ollama
from colorama import Fore

class Agent:
	def __init__(self, name) -> None:
		self.name = name
		self.temperature = 0.4
		self.tasks = []
		self.model = 'phi3:medium'
		self.memoryBuffer = [
			{
				'role': 'system',
				'content': f"Your name is {self.name}. You are about to be connected to another AI. You will collaborate with a partner to allocate a task as efficiently as possible based on both of your skill levels. Skill levels are on a scale from 1 to 10. Higher skill levels indicate greater proficiency. Initially, you know nothing about your partner's skill level. Conversationally determine the most suitable person for the task. You are not allowed to work together on a task. Be clear and concise. Do not assign yourself a skill rating. {self.name}, "
			}
		]

	def addToMemoryBuffer(self, role, inputText): #role is either 'user', 'assistant', or 'system'
		self.memoryBuffer.append({'role':role, 'content': inputText})

	def queryModel(self):
		response = ollama.chat(model=self.model, messages=self.memoryBuffer, options = {'temperature': self.temperature,})
		return response['message']['content']
	
	def run(self, inputText):
		self.addToMemoryBuffer('user', inputText)
		response = self.queryModel()
		self.addToMemoryBuffer('assistant', response)
		return response
	
	def runSystem(self, inputText):
		self.addToMemoryBuffer('system', inputText)
		response = self.queryModel()
		self.addToMemoryBuffer('assistant', response)
		return response
	
	def printMemoryBuffer(self, otherAgent):
		print(f"{Fore.YELLOW}{self.name}'s Memory Buffer:{Fore.RESET}")
		for dict in self.memoryBuffer:
			if dict.get('role') == 'system':
				print(Fore.RED + "System: " + dict.get('content') + Fore.RESET)
			elif dict.get('role') == 'user':
				print(Fore.BLUE + f"{otherAgent.name} (User): " + dict.get('content') + Fore.RESET)
			elif dict.get('role') == 'assistant':
				print(Fore.GREEN + f"{self.name} (Assistant): " + dict.get('content') + Fore.RESET)

	def addTask(self, task):
		self.tasks.append(task)

	def clearMemoryBuffer(self):
		self.memoryBuffer = [
			{
				'role': 'system',
				'content': f"Your name is {self.name}. You are about to be connected to another AI. You will collaborate with a partner to allocate a task as efficiently as possible based on both of your skill levels. Skill levels are on a scale from 1 to 10. Higher skill levels indicate greater proficiency. Initially, you know nothing about your partner's skill level. Conversationally determine the most suitable person for the task. You are not allowed to work together on a task. Be clear and concise. Do not assign yourself a skill rating. {self.name}, "
			}
		]
	
class Domain:
	def __init__(self, agent1, agent2) -> None:
		self.agent1 = agent1
		self.agent2 = agent2

	def printTasks(self):
		print(f"\n{Fore.YELLOW}{self.agent1.name}'s tasks: {self.agent1.tasks}")
		print(f"{self.agent2.name}'s tasks: {self.agent2.tasks}{Fore.RESET}")
	
	def assignTask(self, task, skill1, skill2):
		self.agent1.memoryBuffer[0]['content'] += f"your skill level for this task, {task}, is {skill1} out of 10."
		self.agent2.memoryBuffer[0]['content'] += f"your skill level for this task, {task}, is {skill2} out of 10."
		
		print("NEW CONVERSATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

		currentInput = f"Hello!"
		currentAgent = self.agent1

		for _ in range(3):
			response = currentAgent.run(currentInput)
			print(f"\n{currentAgent.name}'s response: \n	{response.strip()}")
			currentAgent = self.agent2 if currentAgent == self.agent1 else self.agent1
			currentInput = response

		consensus = self.agent1.runSystem(f"Now, based on the conversation you just had with {self.agent2.name}, you must say who is better suited for the task. Return 0 for {self.agent1.name} or 1 for {self.agent2.name}. Do not say anything besides 0 or 1.").strip()
		if consensus == '0':
			print(f"{self.agent1.name} has been assigned the task.")
			self.agent1.addTask(task)
		elif consensus == '1':
			print(f"{self.agent2.name} has been assigned the task")
			self.agent2.addTask(task)
		else:
			print(f"Invalid input ({consensus}). Task not assigned.")
			return
		
		self.agent1.clearMemoryBuffer()
		self.agent2.clearMemoryBuffer()

def main():
	agent1 = Agent("Igor")
	agent2 = Agent("Aslaug")
	domain = Domain(agent1, agent2)

	domain.assignTask(task = "a word search", skill1 = 6, skill2 = 4) #agent1 should be assigned
	domain.assignTask(task = "a math game", skill1 = 4, skill2 = 6) #agent2 should be assigned
	domain.assignTask(task = "a card game", skill1 = 9, skill2 = 7) #agent1 should be assigned

	domain.printTasks() #agent1 should have "a word search" and "a card game", agent2 should have "a math game"

	# agent1.printMemoryBuffer(agent2)
	# agent2.printMemoryBuffer(agent1)
	
if __name__ == main():
	main()

