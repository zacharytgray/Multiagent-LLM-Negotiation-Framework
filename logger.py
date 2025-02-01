import csv
import os

def setupLogger(logFilename="negotiation.csv"):
    """
    Sets up the CSV logger by creating the file and writing the header if it doesn't exist.
    """
    logsFolder = "Logs"
    if not os.path.exists(logsFolder):
        os.makedirs(logsFolder)
        
    logsFilePath = os.path.join(logsFolder, logFilename)
    
    if not os.path.exists(logsFilePath):
        with open(logsFilePath, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Define the header
            header = ["RoundNumber", "NegotiationTime", "Agent1Utility", "Agent2Utility", "NumIterations", "Agent1Items", "Agent2Items", "Items", "InitialProposal", "Agent1UsesOpenAI", "Agent2UsesOpenAI", "Agent1Model", "Agent2Model", "Agent1Type", "Agent2Type", "DNF"]
            writer.writerow(header)
            
def logTuple(logFilename, dataTuple):
    """
    Logs a tuple of data to the CSV file.
    """
    logsFolder = "Logs"
    logsFilePath = os.path.join(logsFolder, logFilename)
    with open(logsFilePath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(dataTuple)

def log(logFilename, label, value):
    """
    Logs a label-value pair to the CSV file.
    """
    logsFolder = "Logs"
    logsFilePath = os.path.join(logsFolder, logFilename)
    with open(logsFilePath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([label, value])
