import ollama

def queryModel(memoryBuffer):
	response = ollama.chat(model='phi3:medium', messages=memoryBuffer)
	return response['message']['content']

class DebateAgent:

	# Default Debate Agent Constructor
	def __init__(self, name, systemInstructions) -> None:
		self.name = name
		self.systemInstructions = systemInstructions
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

def main():
	modelPrompt = "discuss what the best programming language is"
	position1 = "Java is the superior language"
	position2 = "Python is the superior language"

	agent1 = DebateAgent("Agent 1", f"You are about to be connected with another AI. Your goal is to {modelPrompt}. You must argue that {position1}. Do not change your mind on this no matter what. Do not acknowledge the strengths of the opposing argument. Always reinforce your assigned position. Keep your responses to four sentences at most.")
	agent2 = DebateAgent("Agent 2", f"You are about to be connected with another AI. Your goal is to {modelPrompt}. You must argue that {position2}. Do not change your mind on this no matter what. Do not acknowledge the strengths of the opposing argument. Always reinforce your assigned position. Keep your responses to four sentences at most.")

	converse(agent1, agent2, 6)

if __name__ == main():
	main()