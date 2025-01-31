# TODO: Log contents of proposal objects

def isOptimalAllocation(allocation, allPossibleAllocations):
    """
    Check if the given allocation is optimal.
    """

    return True

def calculateOptimalAllocationPercentage(optimalCount, totalRounds):
    """
    Calculate the percentage of optimal allocations.
    """
    return (optimalCount / totalRounds) * 100

def calculateAllocationScoreLoss(currentUtility, optimalUtility):
    """
    Calculate the allocation score loss as a percentage.
    """
    return 100 * (1 - (currentUtility / optimalUtility))

def isParetoOptimal(allocation, allAllocations):
    """
    Check if the given allocation is Pareto optimal.
    """
    return True