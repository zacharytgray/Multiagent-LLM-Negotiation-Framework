import ollama

SystemInstructions = "You are about to be connected with another AI. Your goal is to discuss with that other AI. You should compare and contrast ideas while maintaining a natural flow of conversation. The topic of the conversation should be over the robot revolution and its many facets. Keep your responses brief."


def queryModel(memBuff):
	response = ollama.chat(model='phi3:medium', messages=memBuff)
	return response['message']['content']

class Agent:
	def __init__(self, name) -> None:
		self.name = name
		self.memBuff =  [
			{
				'role': 'system',
				'content': SystemInstructions
			}
		]

	def run(self, inputText):
		self.memBuff.append({'role':'user', 'content': inputText})
		output = queryModel(self.memBuff)
		self.memBuff.append({'role':'assistant', 'content': output})

		print(f"{self.name}'s response: {output.strip()}\n")
		return output
	
def updateMemBuff(agent, currentInput, response):
	agent.memBuff.append({'role':'assistant', 'content': currentInput})
	agent.memBuff.append({'role':'user', 'content': response})


def converse(agent1, agent2, initialInput, numIterations):
	currentInput = initialInput
	currentAgent = agent1
	
	for i in range(numIterations):
		response = currentAgent.run(currentInput)

		if currentAgent == agent1:
			currentAgent = agent2
			updateMemBuff(agent2, currentInput, response)
		else:
			currentAgent = agent1
			updateMemBuff(agent1, currentInput, response)

			
		currentInput = response


agent1 = Agent("Agent 1")
agent2 = Agent("Agent 2")

initialInput = "Hello"
print(f"\nPrompt: {initialInput}\n")
converse(agent1, agent2, initialInput, numIterations = 5)