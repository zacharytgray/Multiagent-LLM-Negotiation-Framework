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
				'content': f"Your name is {self.name}. You are about to be connected to another AI. You will collaborate with a partner to allocate a task as efficiently as possible based on both of your skill levels. Skill levels are on a scale from 1 to 10. Higher skill levels indicate greater proficiency. Initially, you know nothing about your partner's skill level. Conversationally determine the most suitable person for the task. Be clear and concise. Do not assign yourself a skill rating. {self.name}, "
			}
		]

	def addToMemoryBuffer(self, role, inputText): #role is either 'user', 'assistant', or 'system'
		self.memoryBuffer.append({'role':role, 'content': inputText})

	def queryModel(self, inputText):
		# self.memoryBuffer.append({'role':'user', 'content': inputText})
		self.addToMemoryBuffer('user', inputText)
		response = ollama.chat(model=self.model, messages=self.memoryBuffer, options = {'temperature': self.temperature,})
		# self.memoryBuffer.append({'role':'assistant', 'content': response['message']['content']})
		self.addToMemoryBuffer('assistant', response['message']['content'])
		return response['message']['content']
	
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
	
class Domain:
	def __init__(self, agent1, agent2) -> None:
		self.agent1 = agent1
		self.agent2 = agent2
	
	def assignTask(self, task, skill1, skill2):
		self.agent1.memoryBuffer[0]['content'] += f"your skill level for this task, {task}, is {skill1} out of 10."
		self.agent2.memoryBuffer[0]['content'] += f"your skill level for this task, {task}, is {skill2} out of 10."
		
		currentInput = f"Hello!"
		currentAgent = self.agent1

		for _ in range(3):
			response = currentAgent.queryModel(currentInput)
			print(f"\n{currentAgent.name}'s response: \n	{response.strip()}")
			currentAgent = self.agent2 if currentAgent == self.agent1 else self.agent1
			currentInput = response

def main():
	agent1 = Agent("Igor")
	agent2 = Agent("Aslaug")
	domain = Domain(agent1, agent2)

	task = "a word search"
	skill1 = 10
	skill2 = 1

	domain.assignTask(task, skill1, skill2)

	agent1.printMemoryBuffer(agent2)
	agent2.printMemoryBuffer(agent1)
	
if __name__ == main():
	main()

