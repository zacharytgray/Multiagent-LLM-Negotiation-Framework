import ollama

def queryModel(memoryBuffer):
	response = ollama.chat(model='phi3:medium', messages=memoryBuffer)
	return response['message']['content']

class Agent:
	def __init__(self, name, systemInstructions) -> None:
		self.name = name
		self.memoryBuffer =  [
			{
				'role': 'system',
				'content': systemInstructions
			}
		]

	def run(self, inputText):
		self.memoryBuffer.append({'role':'user', 'content': inputText})
		output = queryModel(self.memoryBuffer)
		self.memoryBuffer.append({'role':'assistant', 'content': output})

		print(f"\n{self.name}'s response: \n	{output.strip()}")
		return output

def converse(agent1, agent2, numIterations):
	currentInput = "Hello"
	currentAgent = agent1
	
	for _ in range(numIterations):
		response = currentAgent.run(currentInput)
		currentAgent = agent2 if currentAgent == agent1 else agent1
		currentInput = response

def debate():
	modelPrompt = "discuss what the best programming language is"
	position1 = "Java is the superior language"
	position2 = "Python is the superior language"

	agent1 = Agent("Agent 1", f"You are about to be connected with another AI. Your goal is to {modelPrompt}. You must argue that {position1}. Do not change your mind on this no matter what. Do not acknowledge the strengths of the opposing argument. Always reinforce your assigned position. Keep your responses to four sentences at most.")
	agent2 = Agent("Agent 2", f"You are about to be connected with another AI. Your goal is to {modelPrompt}. You must argue that {position2}. Do not change your mind on this no matter what. Do not acknowledge the strengths of the opposing argument. Always reinforce your assigned position. Keep your responses to four sentences at most.")

	converse(agent1, agent2, 6)

def taskAllocation():
	agent1task1 = "3/10"
	agent1task2 = "9/10"

	agent2task1 = "1/10"
	agent2task2 = "1/10"

	task1 = "A recognizing game"
	task2 = "A word hunt game"

	agent1Instructions = f"You are about to be connected with another AI. Your goal is to allocate two tasks between the two of you based on your own expertise and weaknesses for both tasks. You must negotiate with the other AI to complete this, learning their strengths and weaknesses conversationally. Your skill level is on a scale of 1 to 10, where 10 is used for tasks you're very confident in, and 1 is used for those you are not confident in whatsoever. Respond in no more than four sentences. Your unique skill level for Task 1, {task1}, is {agent1task1}, and your unique skill level for Task 2, {task2}, is {agent1task2}. Note that the other AI likely has different skill levels than you. Do not deviate from these given skill levels."
	agent2Instructions = f"You are about to be connected with another AI. Your goal is to allocate two tasks between the two of you based on your own expertise and weaknesses for both tasks. You must negotiate with the other AI to complete this, learning their strengths and weaknesses conversationally. Your skill level is on a scale of 1 to 10, where 10 is used for tasks you're very confident in, and 1 is used for those you are not confident in whatsoever. Respond in no more than four sentences. Your unique skill level for Task 1, {task1}, is {agent2task1}, and your unique skill level for Task 2, {task2}, is {agent2task2}. Note that the other AI likely has different skill levels than you. Do not deviate from these given skill levels."

	agent1 = Agent("Agent 1", agent1Instructions)
	agent2 = Agent("Agent 2", agent2Instructions)

	converse(agent1, agent2, 6)

	print("\nAGENT 1 MEMORY BUFFER:\n		")
	print(agent1.memoryBuffer)
	print("\nAGENT 2 MEMORY BUFFER:\n		")
	print(agent2.memoryBuffer)

def main():
	taskAllocation()

if __name__ == main():
	main()