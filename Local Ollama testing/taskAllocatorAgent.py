import ollama

def queryModel(memoryBuffer):
	response = ollama.chat(model='phi3:medium', messages=memoryBuffer)
	return response['message']['content']

class TaskAllocatorAgent:
	# Agent Constructor used for Task Allocation
	def __init__(self, name, systemInstructions, task1, task2, skill1, skill2) -> None:
		self.name = name
		self.task1 = task1
		self.task2 = task2
		self.skill1 = skill1
		self.skill2 = skill2
		self.memoryBuffer = [
			{
				'role': 'system',
				'content': systemInstructions
			},
			{
				'role': 'system',
				'content': f"You should always remember your skill levels. Task 1 ({task1}): {skill1}, Task 2 ({task2}): {skill2}."
			}
		]

		print(f"{self.name} Skill Levels: Task 1 ({self.task1}): {self.skill1}, Task 2 ({self.task2}): {self.skill2}")


	def run(self, inputText):
		reminderStr = f" You should always remember your skill levels. Task 1 ({self.task1}): {self.skill1}, Task 2 ({self.task2}): {self.skill2}."
		# self.memoryBuffer.append({'role':'system', 'content': reminderStr})
		self.memoryBuffer.append({'role':'user', 'content': inputText})
		output = queryModel(self.memoryBuffer)
		self.memoryBuffer.append({'role':'assistant', 'content': output})

		print(f"\n{self.name}'s response: \n	{output.strip()}")
		return output

def converse(agent1, agent2, numIterations):
	currentInput = "Hello! Let's begin the task allocation. "
	currentAgent = agent1
	
	for _ in range(numIterations):
		response = currentAgent.run(currentInput)
		currentAgent = agent2 if currentAgent == agent1 else agent1
		currentInput = response


def taskAllocation():
	agent1skill1 = "3/10"
	agent1skill2 = "9/10"

	agent2skill1 = "10/10"
	agent2skill2 = "4/10"

	task1 = "a geography game"
	task2 = "a word game"

	systemInstructions = "You are about to be connected with another AI. Your goal is to allocate two tasks between the two of you based on your own expertise and weaknesses for both tasks. You must negotiate with the other AI to complete this, learning their strengths and weaknesses conversationally. Your skill level is on a scale of 1 to 10, where 10 is used for tasks you're very confident in, and 1 is used for those you are not confident in whatsoever. Respond in no more than four sentences. Note that the other AI likely has different skill levels than you. Do not deviate from your given skill levels. "

	agent1Instructions = f"Your unique skill level for Task 1, {task1}, is {agent1skill1}, and your unique skill level for Task 2, {task2}, is {agent1skill2}."
	agent2Instructions = f"Your unique skill level for Task 1, {task1}, is {agent2skill1}, and your unique skill level for Task 2, {task2}, is {agent2skill2}."

	agent1 = TaskAllocatorAgent("Agent 1", systemInstructions + agent1Instructions, task1, task2, agent1skill1, agent1skill2)
	agent2 = TaskAllocatorAgent("Agent 2", systemInstructions + agent2Instructions,  task1, task2, agent2skill1, agent2skill2)

	converse(agent1, agent2, 3)

	print("\nAGENT 1 MEMORY BUFFER:\n		")
	print(agent1.memoryBuffer)
	print("\nAGENT 2 MEMORY BUFFER:\n		")
	print(agent2.memoryBuffer)

def main():
	taskAllocation()

if __name__ == main():
	main()