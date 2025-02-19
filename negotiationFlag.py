
from enum import Enum

class NegotiationFlag(Enum):
    PROPOSAL_NOT_FOUND = "Proposal not found in response"
    NOT_ENOUGH_TASKS = "Not enough tasks in proposal"
    TOO_MANY_TASKS = "Too many tasks in proposal"
    INVALID_TASKS_PRESENT = "Invalid tasks present in proposal"
    ERROR_FREE = "No errors found"
    INVALID_PROPOSAL_FORMAT = "Invalid proposal format"
    INVALID_AGENT_NAME = "Invalid agent name in proposal"
    TIMEOUTERROR = "Timeout error"
    PROPOSAL_MISMATCH = "Proposal mismatch detected"