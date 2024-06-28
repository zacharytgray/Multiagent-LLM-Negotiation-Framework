import ollama
from colorama import Fore

class Agent:
	def __init__(self, items) -> None:
		self.temperature = 0.4
		self.items = items
		self.model = 'phi3:medium'
		self.memoryBuffer = [
			{
				'role': 'system',
				'content': f"You are a grocery store clerk. You have many items in inventory, and you must answer questions about these items. Do not make up items that do not exist in your inventory. Your inventory is as follows: {self.items}."
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
	

def main():
	agent1 = Agent(["$2 Apple (12)", "$3 Orange (10)", "$25 Banana (8)"])
	agent2 = Agent(["$4 Apple (32)", "$5 Orange (7)", "$68 Banana (90)"])

	userInput = ""
	while userInput.lower() != "q":
		userInput = input("You: ")
		if userInput != "q":
			response = agent1.run(userInput)
			print(f"Assistant 1: {response}")

			response = agent2.run(userInput)
			print(f"Assistant 2: {response}")
		else:
			print("Goodbye!")

	userInput = "What is the price of the apples?"
	print(agent1.run(userInput))
	print(agent2.run(userInput))

if __name__ == main():
	main()