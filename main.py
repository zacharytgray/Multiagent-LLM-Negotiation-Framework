from negotiation import Negotiation
from logger import setupLogger, log, logTuple
import datetime
    
def main():
    
    # TODO: Add deepseek support
    # TODO: Store logs in Logs folder
    # TODO: Change .log to csv files
    
    #Test Parameters
    numRounds = 1
    numTasks = 6
    maxIterations = 32
    
    agent1Name = "Finn"
    agent1Model = "gemma2:latest"
    agent1usesOpenAI = False
    agent1Type = "default"
    
    agent2Name = "Jake"
    agent2Model = "gemma2:latest"
    agent2usesOpenAI = False
    agent2Type = "default"
    
    # Setup csv logger
    logFilename = constructLogFilename(agent1Model, agent2Model)
    setupLogger(logFilename=logFilename)
    
    log(logFilename, "NumTasks", numTasks) # Label, value
    log(logFilename, "Agent1Model", agent1Model) 
    log(logFilename, "Agent2Model", agent2Model) 

    negotiationStartTime = datetime.datetime.now()
    # Run the negotiation rounds
    for roundIndex in range(1, numRounds + 1):
        n = Negotiation(roundIndex, numTasks, maxIterations, agent1Model, agent1usesOpenAI, agent1Type, agent2Model, agent2usesOpenAI, agent2Type, agent1Name, agent2Name)
        n.startNegotiation()
        dataTuple = (n.roundIndex, 
            n.negotiationTime,
            n.winningProposal.agent1Utility, 
            n.winningProposal.agent2Utility,        
            n.numIterations,   
            n.winningProposal.agent1Tasks, 
            n.winningProposal.agent2Tasks, 
            n.tasks,)
        logTuple(logFilename, dataTuple)
    totalNegotiationTime = datetime.datetime.now() - negotiationStartTime
    log(logFilename, "TotalNegotiationTime", totalNegotiationTime)
    log(logFilename, "AverageTimePerRound", totalNegotiationTime / numRounds)
    
def constructLogFilename(agent1Model, agent2Model):
    sanitizedAgent1Model = ''.join(filter(str.isalnum, agent1Model))
    sanitizedAgent2Model = ''.join(filter(str.isalnum, agent2Model))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return f"{sanitizedAgent1Model}_{sanitizedAgent2Model}_{timestamp}.csv"

if __name__ == "__main__":
    main()