# Project Report: Autonomous Agent using Q-Learning

## Overview
The goal of this project was to develop an autonomous agent capable of navigating a 2D grid environment with obstacles to reach a target. The solution uses Q-learning, a reinforcement learning technique, to update the agent's decision policy based on environmental feedback.

## Logic and Model
- **Environment:**  
  A grid is used to represent the 2D space with a start point, a defined goal, and several fixed obstacles.
  
- **Agent’s Perception:**  
  The agent’s state is its current grid cell. Additionally, for simulated sensor data, the Euclidean distance and angle from the agent to the goal are computed (these values could be used in more complex models).

- **Actions:**  
  The agent can move in one of four directions: UP, DOWN, LEFT, RIGHT.

- **Reward System:**  
  - **Positive reward** when the goal is reached.
  - **Negative reward** for hitting obstacles.
  - A small **step penalty** encourages the agent to find the shortest path.

- **Learning Process:**  
  Using Q-learning, the agent starts with a Q-table of zeros. Over multiple episodes, the Q-values are updated based on the received reward and the estimated future rewards. An epsilon-greedy strategy ensures exploration of the environment.

## Challenges Faced
- **Balancing Exploration vs. Exploitation:**  
  Finding the right epsilon value was challenging—too high resulted in random behavior, while too low led to premature convergence.
  
- **State Representation:**  
  Using the grid cell as the state is simple yet effective; however, incorporating sensor-like data (distance and angle) into a more complex representation could lead to better navigation but would require a more complex learning algorithm (e.g., deep reinforcement learning).

## Future Improvements
- **Multiple Agents or Dynamic Obstacles:**  
  Introduce multiple agents or obstacles that move over time.
- **Advanced ML Techniques:**  
  Implement deep Q-learning (DQN) to handle richer state spaces (including sensor inputs) and more continuous actions.
- **Enhanced Visualization and Logging:**  
  Create a dashboard to log more detailed performance metrics in real time.
# vyorius_ml