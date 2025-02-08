import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI 
from langchain_ollama import ChatOllama 
from colorama import Fore
from dotenv import load_dotenv
from negotiationFlag import NegotiationFlag
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
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
        self.systemInstructions = ""
        self.initialProposalHelperInstructions = ""
        self.setUpModel()
        self.loadSystemInstructions()
        self.responseTimeout = 320 # seconds

    def setUpModel(self):
        if self.usesOpenAI: # Assign model based on the model type
            load_dotenv("keys.env")
            self.openaiApiKey = os.getenv("OPENAI_API_KEY")
            if self.openaiApiKey is None:
                raise ValueError("OpenAI API key is not found")
            self.model = ChatOpenAI(model_name=self.modelName, openai_api_key=self.openaiApiKey, temperature=1) 
        else:
            self.model = ChatOllama(model=self.modelName, base_url="http://localhost:11434", temperature=0.1, num_predict=2000)

        self.instructionsFilename = "SystemInstructions/deepseekCollaborativeInstructions.txt"
        self.initialPropHelperFname = "SystemInstructions/initialProposalHelperInstructions.txt"
        
    def loadSystemInstructions(self): # Load system instructions from file
        try:
            with open(self.instructionsFilename, 'r') as file:
                self.systemInstructions += file.read()
            with open(self.initialPropHelperFname, 'r') as file:
                self.initialProposalHelperInstructions = file.read()
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

    async def generateResponseAsync(self, role=None, inputText=None): # Generate response based on input
        try:
            if inputText and role:
                self.addToChatHistory(role, inputText)
                
            history = ChatPromptTemplate.from_messages(self.memory)
            chain = history | self.model
            response = await chain.ainvoke({})
            response_content = response.content if isinstance(response, AIMessage) else response
            
            if self.modelName.lower().startswith("deepseek"):
                #Print the part of the response between the think tags
                capture_group = re.search(r"<think>(.*?)</think>", response_content, re.DOTALL)
                if capture_group:
                    thought_content = capture_group.group(1)
                else:
                    thought_content = response_content
                # print(f"Thoughts: {thought_content}")
                
                pattern = r"<think>.*?</think>"
                response_content = re.sub(pattern, "", response_content, flags=re.DOTALL)
            self.addToChatHistory('assistant', response_content)
            return response_content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return NegotiationFlag.TIMEOUTERROR
                
        
    def generateResponse(self, role=None, inputText=None): # Generate response based on input
        try:
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(
                asyncio.wait_for(
                    self.generateResponseAsync(role, inputText),
                    timeout = self.responseTimeout
                    )
                )
            return response
        except asyncio.TimeoutError:
            print(f"{Fore.RED}Timeout error while generating response for {self.agentName}{Fore.RESET}")
            return NegotiationFlag.TIMEOUTERROR
        
    def clearMemory(self):
        self.memory = self.memory[:1] # Keep the system message
    
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
