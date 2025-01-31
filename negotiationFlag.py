
from enum import Enum

class NegotiationFlag(Enum):
    PROPOSAL_NOT_FOUND = "Proposal not found in response"
    NOT_ENOUGH_ITEMS = "Not enough items in proposal"
    TOO_MANY_ITEMS = "Too many items in proposal"
    INVALID_ITEMS_PRESENT = "Invalid items present in proposal"
    ERROR_FREE = "No errors found"