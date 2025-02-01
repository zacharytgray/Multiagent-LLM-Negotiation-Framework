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
        self.systemInstructions = f"Your name is {self.agentName}.\n"
        self.instructionsFilename = None # Set in setUpModel()
        self.setUpModel()
        self.loadSystemInstructions()

    def setUpModel(self):
        if self.usesOpenAI: # Assign model based on the model type
            load_dotenv("keys.env")
            self.openaiApiKey = os.getenv("OPENAI_API_KEY")
            if self.openaiApiKey is None:
                raise ValueError("OpenAI API key is not found")
            self.model = ChatOpenAI(model_name=self.modelName, openai_api_key=self.openaiApiKey, temperature=0.3) 
        else:
            self.model = ChatOllama(model=self.modelName, base_url="http://localhost:11434", temperature=0.3)
            
        # Set the instructions file based on the model type
        if self.agentType == "default":
            self.instructionsFilename = "SystemInstructions/defaultCollaborativeInstructions.txt"
        elif self.modelName.lower().startswith("deepseek"):
            self.instructionsFilename = "SystemInstructions/deepseekCollaborativeInstructions.txt"
        
    def loadSystemInstructions(self):
        try:
            with open(self.instructionsFilename, 'r') as file:
                self.systemInstructions += file.read()
                self.addToChatHistory('system', self.systemInstructions)
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
        
    def generateResponse(self, role, inputText):
        self.addToChatHistory(role, inputText)
        history = ChatPromptTemplate.from_messages(self.memory)
        chain = history | self.model
        response = chain.invoke({})
        response_content = response.content if isinstance(response, AIMessage) else response
        
        # TODO: If self.modelName.lower() starts with "deepseek", remove from where it says "<think>" at the start of its response to when it says </think> (inclusive). Keep everything after that.
        if self.modelName.lower().startswith("deepseek"):
            pattern = r"<think>.*?</think>"
            response_content = re.sub(pattern, "", response_content, flags=re.DOTALL)
        response_content = response_content.replace('*', '')
        self.addToChatHistory('assistant', response_content)
        return response_content.strip()
    