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
				'content': systemInstructions + f"Your unique skill levels are as follows: Task 1 ({self.task1}) - skill level {self.skill1}, Task 2 ({self.task2}) - skill level {self.skill2}."
			},
		]

		print(f"{self.name} Skill Levels: Task 1 ({self.task1}): {self.skill1}, Task 2 ({self.task2}): {self.skill2}")


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

def converse(agent1, agent2, numIterations):
	currentInput = "Hello! Let's begin the task allocation. "
	currentAgent = agent1
	
	for _ in range(numIterations):
		response = currentAgent.run(currentInput)
		print(f"\n{currentAgent.name}'s response: \n	{response.strip()}")
		currentAgent = agent2 if currentAgent == agent1 else agent1
		currentInput = response
	

def taskAllocation():
	agent1skill1 = "3/10"
	agent1skill2 = "8/10"

	agent2skill1 = "9/10"
	agent2skill2 = "2/10"

	task1 = "a geography game"
	task2 = "a word game"

	systemInstructions = "You are about to be connected with another AI. Your goal is to allocate two tasks between the two of you based on your own unique, pre-assigned strengths and weaknesses for both tasks. You must negotiate with the other AI to complete this, learning their strengths and weaknesses conversationally without any prior knowledge of the other AI's skill levels. Both of your skill levels are predetermined, and are on a scale of 1 to 10, where 10 is used for tasks you're proficient at, and 1 is used for those you cannot complete. Respond in no more than four sentences. Do not change your assigned skill levels. "

	agent1 = TaskAllocatorAgent("Agent 1", systemInstructions, task1, task2, agent1skill1, agent1skill2)
	agent2 = TaskAllocatorAgent("Agent 2", systemInstructions, task1, task2, agent2skill1, agent2skill2)
	
	converse(agent1, agent2, 3)

	# print("\nAGENT 1 MEMORY BUFFER:\n		")
	# print(agent1.memoryBuffer)
	# print("\nAGENT 2 MEMORY BUFFER:\n		")
	# print(agent2.memoryBuffer)

	# Now, assume a consensus has been reached.
	mediatorInstructions = "Now, based on the conversation you just had, you will provide the appropriate task allocation in the following format: {'Agent 1': 'Task X', 'Agent 2': 'Task Y'}, where tasks X and Y are either " + task1 + " or " + task2 + ". Include the brackets in your reponse. Do not reply with anything else. Note that you are Agent 1, and you've been talking to Agent 2."
	consensus = agent1.run(mediatorInstructions)
	print(f"Consensus: {consensus}")

def main():
	taskAllocation()

if __name__ == main():
	main()