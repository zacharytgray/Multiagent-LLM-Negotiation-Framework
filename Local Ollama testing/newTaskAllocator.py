import ollama
from colorama import Fore

class Agent:
	def __init__(self, name) -> None:
		self.name = name
		self.temperature = 0.4
		self.tasks = []
		self.memoryBuffer = [
			{
				'role': 'system',
				'content' : f"Your name is {name}. You are about to be connected with another Agent. Your goal is to most optimally allocate tasks between the two of you based on your assigned skill levels for each task. Higher skill levels indicate higher proficiency for that particular task. You don't know anything about the other agent initially, so you must compare your own strengths and weaknesses conversationally to determine who gets to perform a given task. You will receive one task at a time. Make sure to mention your corresponding skill level for a task in your reponse. Keep your responses brief."
			}
		]

	def queryModel(self, inputText):
		self.memoryBuffer.append({'role':'user', 'content': inputText})
		response = ollama.chat(model='phi3:medium', messages=self.memoryBuffer, options = {'temperature': self.temperature,})
		self.memoryBuffer.append({'role':'assistant', 'content': response['message']['content']})
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
	
	def assignTask(self, skill1, skill2, task):
		agent1Instructions = f"The next task you must allocate between the two of you is {task}. Your skill level for this task is {skill1}. Begin the task allocation for this task."
		agent2Instructions = f"The next task you must allocate between the two of you is {task}. Your skill level for this task is {skill2}. Begin the task allocation for this task."

		self.agent1.memoryBuffer.append({'role':'system', 'content': agent1Instructions})
		self.agent2.memoryBuffer.append({'role':'system', 'content': agent2Instructions})

		currentInput = "Hello! Let's begin allocating this task."
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
	skill1 = 8
	skill2 = 3

	domain.assignTask(skill1, skill2, task)

	agent1.printMemoryBuffer(agent2)
	agent2.printMemoryBuffer(agent1)
	
if __name__ == main():
	main()

