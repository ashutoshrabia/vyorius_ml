# 2D Autonomous Agent with Q-Learning

## Overview
This project builds a basic autonomous agent that navigates a grid-based 2D environment towards a goal while avoiding obstacles. The agent learns to navigate using Q-learning, a reinforcement learning technique, instead of using simple rule-based logic.

## Setup Instructions
1. **Install Python 3.6 or later.**
2. **Install dependencies:**  
   Run the following command to install the required libraries:
-> pip install pygame numpy
3. **Run the agent:**  
Execute the script using:


## ML Technique Used
This solution uses **Q-learning**:
- **State:** Agent's grid location (x, y).  
- **Actions:** Up, Down, Left, Right.
- **Reward Structure:**  
- +100 for reaching the goal.
- -10 for hitting an obstacle.
- -1 for each move to encourage efficient paths.
- **Learning:** The Q-table is updated using the Q-learning update rule.
