import ollama
from colorama import Fore

def queryModel(memoryBuffer):
	response = ollama.chat(model='phi3:medium', messages=memoryBuffer, options = {'temperature': 0.4,})
	return response['message']['content']

def printMemoryBuffer(agent):
	for dict in agent.memoryBuffer:
		if dict.get('role') == 'system':
			print(Fore.RED + "SYSTEM: " + dict.get('content') + Fore.RESET)
		elif dict.get('role') == 'user':
			print(Fore.BLUE + "AGENT 2 (USER): " + dict.get('content') + Fore.RESET)
		elif dict.get('role') == 'assistant':
			print(Fore.GREEN + "AGENT 1 (ASSISTANT): " + dict.get('content') + Fore.RESET)

class TaskAllocatorAgent:
	# Agent Constructor used for Task Allocation
	def __init__(self, name, task1, task2, skill1, skill2) -> None:
		self.name = name
		self.task1 = task1
		self.task2 = task2
		self.skill1 = skill1
		self.skill2 = skill2
		self.memoryBuffer = [
			{
				'role': 'system',
				'content': f"You are about to be connected with another Agent. Your goal is to allocate two tasks between the two of you as efficiently as possible. For each task, you are given unique skill levels. Task 1 is {task1}, and Task 2 is {task2}. Your skill level for Task 1 is {skill1}, and your skill level for Task 2 is {skill2}. These skill levels are on a scale of 1 to 10, where 10 is the best at performing a task, and 1 is no ability to perform the task. The other agent has their own, unique skill levels that you will need to discover through conversation. Note that it is possible for an agent to take on both tasks if their scores are better for both tasks. Negotiate the most optimal and efficient task allocation between the two of you. Whenever you feel like the two of you have agreed on a solution, include the word 'EXIT' in your response. Do not use the word EXIT at all until you are absolutely sure the conversation should end."
				# 'content': f"You are {name}. Your goal is to allocate two tasks as efficiently as possible. Task 1 is {task1}, and Task 2 is {task2}. Your skill level for Task 1 is {skill1}, and your skill level for Task 2 is {skill2}. These skill levels are on a scale of 1 to 10. Discover the other agent's skill levels through conversation. The other agent has their own skill levels. Include the word 'EXIT' when you agree on a solution."
			},
		]

		print(f"{Fore.YELLOW}{self.name} Skill Levels: Task 1 ({self.task1}): {self.skill1}, Task 2 ({self.task2}): {self.skill2}{Fore.RESET}")


	def run(self, inputText):
		# reminderStr = f" You should always remember your skill levels. Your Task 1 ({self.task1}) skill level is {self.skill1}, and your Task 2 ({self.task2}) skill level is {self.skill2}."
		self.memoryBuffer.append({'role':'user', 'content': inputText})
		# self.memoryBuffer.append({'role':'system', 'content': reminderStr})

		try:
			output = queryModel(self.memoryBuffer)
			if not output.strip(): #Check for empty response
				raise ValueError(f"Received an empty response from {self.name}")
		except Exception as e:
			print(f"Error: {e}")
			output = "Sorry, I didn't get that. Can you please repeat?"

		# print(f"\n{self.name}'s response: \n	{output.strip()}")
		self.memoryBuffer.append({'role':'assistant', 'content': output})
		return output

# def allocateTask(agent1, agent2, skill1, skill2, task):

def converse(agent1, agent2):
	currentInput = "Hello! Let's begin the task allocation. "
	currentAgent = agent1
	numIterations = 0

	exit = False
	while not exit:
	# for _ in range(numIterations): # For debugging. Remember to set numIterations if this line is uncommented
		response = currentAgent.run(currentInput)
		print(f"\n{currentAgent.name}'s response: \n	{response.strip()}")
		currentAgent = agent2 if currentAgent == agent1 else agent1
		currentInput = response

		#Check if either agent has ended the conversation
		if 'EXIT' in agent1.memoryBuffer[len(agent1.memoryBuffer)-1].get('content'):
			exit = True
		elif 'EXIT' in agent2.memoryBuffer[len(agent2.memoryBuffer)-1].get('content') and numIterations >= 1:
			exit = True
		numIterations += 1

def taskAllocation():
	agent1skill1 = "2"
	agent1skill2 = "3"

	agent2skill1 = "6"
	agent2skill2 = "9"

	task1 = "a geography game"
	task2 = "a word game"

	agent1 = TaskAllocatorAgent("Agent 1", task1, task2, agent1skill1, agent1skill2)
	agent2 = TaskAllocatorAgent("Agent 2", task1, task2, agent2skill1, agent2skill2)
	
	converse(agent1, agent2)

	# Print Memory Buffers 
	print(f"\n{Fore.YELLOW}AGENT 1 MEMORY BUFFER:")
	printMemoryBuffer(agent1)
	print(f"\n{Fore.YELLOW}AGENT 2 MEMORY BUFFER:")
	printMemoryBuffer(agent2)

	# Assuming consensus has been reached, we ask Agent 1 to return a dictionary of the assigned tasks.
	mediatorInstructions = "Now, based on the conversation you just had, you will provide the appropriate task allocation in the following format: {'Agent 1': 'Task X', 'Agent 2': 'Task Y'}, where tasks X and Y are either " + task1 + " or " + task2 + ". Include the brackets in your reponse. Do not reply with anything else. Note that you are Agent 1, and you've been talking to Agent 2. If one agent ends up with both tasks, format it like this: {'Agent Z': '" + task1 + " and " + task2 + "'}"
	consensus = agent1.run(mediatorInstructions)
	print(f"\n{Fore.YELLOW}Consensus: {consensus} {Fore.RESET}")

def main():
	taskAllocation()

if __name__ == main():
	main()