import ollama

modelPrompt = "discuss what the best programming language is."
position1 = "Java is the superior language"
position2 = "Python is the superior language"

def queryModel(memBuff):
	response = ollama.chat(model='phi3:medium', messages=memBuff)
	return response['message']['content']

class Agent:
	def __init__(self, name) -> None:
		self.name = name
		self.memBuff =  [
			{
				'role': 'system',
				'content': "You are about to be connected with another AI. Your goal is to discuss with that other AI. You should compare and contrast ideas while maintaining a natural flow of conversation. The topic of the conversation should be over the robot revolution and its many facets. Keep your responses brief."
			}
		]

	def __init__(self, name, customInstructions) -> None:
		self.name = name
		self.memBuff =  [
			{
				'role': 'system',
				'content': customInstructions
			}
		]

	def run(self, inputText):
		self.memBuff.append({'role':'user', 'content': inputText})
		output = queryModel(self.memBuff)
		self.memBuff.append({'role':'assistant', 'content': output})

		print(f"\n{self.name}'s response: \n	{output.strip()}")
		return output


def converse(agent1, agent2, numIterations):
	currentInput = "Hello"
	currentAgent = agent1
	
	for i in range(numIterations):
		response = currentAgent.run(currentInput)

		if currentAgent == agent1:
			currentAgent = agent2
		else:
			currentAgent = agent1
			
		currentInput = response

agent1 = Agent("Agent 1", f"You are about to be connected with another AI. Your goal is to {modelPrompt}. You must argue that {position1}. Do not change your mind on this no matter what. Keep your responses to four sentences at most.")
agent2 = Agent("Agent 2", f"You are about to be connected with another AI. Your goal is to {modelPrompt}. You must argue that {position2}. Do not change your mind on this no matter what. Keep your responses to four sentences at most.")

converse(agent1, agent2, numIterations = 6)