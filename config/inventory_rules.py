"""
BullwhipAI
Inventory Configuration
"""

# Number of days we want inventory coverage for
TARGET_COVERAGE_DAYS = 7

# Safety stock percentage
SAFETY_STOCK_PERCENT = 0.15

# Approximate storage cost
HOLDING_COST_PER_UNIT = 12

# Desired service level
SERVICE_LEVEL = 0.95

# Inventory Health thresholds

HEALTHY_RATIO_LOW = 0.9
HEALTHY_RATIO_HIGH = 1.2