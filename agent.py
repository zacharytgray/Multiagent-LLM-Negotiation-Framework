import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI 
from langchain_ollama import ChatOllama 
from colorama import Fore
from dotenv import load_dotenv
import re

class Agent:
    
    def __init__(self, agentName, modelName, usesOpenAI, agentType):
        self.agentName = agentName
        self.modelName = modelName
        self.usesOpenAI = usesOpenAI
        self.openaiApiKey = None
        self.agentType = agentType
        self.numTokensGenerated = 0 
        self.memory = []
        self.model = None
        self.proposal = None
        self.systemInstructions = f"Your name is {self.agentName}. You are a collaborative agent."
        self.setUpModel()
        self.loadSystemInstructions()

    def setUpModel(self):
        if self.usesOpenAI: # Assign model based on the model type
            load_dotenv("keys.env")
            self.openaiApiKey = os.getenv("OPENAI_API_KEY")
            if self.openaiApiKey is None:
                raise ValueError("OpenAI API key is not found")
            self.model = ChatOpenAI(model_name=self.modelName, openai_api_key=self.openaiApiKey, temperature=1) 
        else:
            self.model = ChatOllama(model=self.modelName, base_url="http://localhost:11434", temperature=0.3)
            
        # Set the instructions file based on the model type
        # if self.agentType == "default":
        #     self.instructionsFilename = "SystemInstructions/defaultCollaborativeInstructions.txt"
            
        # if self.modelName.lower().startswith("deepseek"):
        #     print(f"{Fore.YELLOW}Using DeepSeek instructions{Fore.RESET}")
        #     self.instructionsFilename = "SystemInstructions/deepseekCollaborativeInstructions.txt"
        self.instructionsFilename = "SystemInstructions/deepseekCollaborativeInstructions_JSON.txt"
        
    def loadSystemInstructions(self): # Load system instructions from file
        try:
            with open(self.instructionsFilename, 'r') as file:
                self.systemInstructions += file.read()
        except FileNotFoundError:
            print(f"{Fore.RED}Instructions file not found: {self.instructionsFilename}{Fore.RESET}")
            exit(1)
        
    def addToChatHistory(self, role, content):
        if role == 'system':
            self.memory.append(SystemMessage(content=content))
        elif role == 'user':
            self.memory.append(HumanMessage(content=content))
        elif role == 'assistant':
            self.memory.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unknown role: {role}") 
        
#         if self.modelName.lower().startswith("deepseek") and not isinstance(self.memory[len(self.memory) - 1], SystemMessage):
#             formatReminder = """
# **IMPORTANT REMINDER**: 
# If you want to propose a deal, you MUST use the following format, beginning with AND INCLUDING the 'PROPOSAL:' keyword:

# ... text leading up to proposal ...
# PROPOSAL:
# Your Name: task1, task2, task3, ...
# Partner's Name: task4, task5, task6, ...

# YOU MUST include the keyword "PROPOSAL:" exactly as shown above to make a proposal.
# DO NOT include any other text or punctuation.
# DO NOT include proposals in any other format.
# ONLY USE THE ABOVE FORMAT TO MAKE PROPOSALS.

# **IMPORTANT REMINDER**:
# If you're ready to finalize a deal, you must both say "DEAL!" consecutively. Do not use the word "DEAL!" in any other context."""
#             self.memory.append(SystemMessage(content=formatReminder))
        
    def generateResponse(self, role, inputText):
        self.addToChatHistory(role, inputText)
        history = ChatPromptTemplate.from_messages(self.memory)
        chain = history | self.model
        response = chain.invoke({})
        response_content = response.content if isinstance(response, AIMessage) else response
        
        if self.modelName.lower().startswith("deepseek"):
            pattern = r"<think>.*?</think>"
            response_content = re.sub(pattern, "", response_content, flags=re.DOTALL)
        response_content = response_content.replace('*', '') # Remove asterisks
        response_content = response_content.replace('- ', '') # Remove bullets
        response_content = re.sub(r'(?:\n\s*){2,}', '\n', response_content)
        self.addToChatHistory('assistant', response_content)
        return response_content.strip()
    
    def printMemory(self):
        print(f"----------------{Fore.LIGHTYELLOW_EX}{self.agentName}'s Memory:{Fore.RESET}----------------")
        for i, message in enumerate(self.memory):
            if i == 0: # Skip the system message
                continue
            if isinstance(message, SystemMessage):
                print(f"{Fore.LIGHTRED_EX}System: {message.content}{Fore.RESET}")
            elif isinstance(message, HumanMessage):
                print(f"{Fore.LIGHTGREEN_EX}Partner: {message.content}{Fore.RESET}")
            elif isinstance(message, AIMessage):
                print(f"{Fore.LIGHTBLUE_EX}{self.agentName}: {message.content}{Fore.RESET}")
            else:
                print(f"Unknown message type: {message}")
            print("----------------------------------------------------------------------------------------")
        print(f"----------------{Fore.LIGHTYELLOW_EX}END Memory:{Fore.RESET}----------------")
