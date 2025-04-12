import pygame
import numpy as np
import random
import math
import sys


# Environment Settings
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 30
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10

# RGB
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Q-Learning parameters

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.3       
EPISODES = 500      
MAX_STEPS = 200    

# Environment Class
class GridWorld:
    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.cell_size = CELL_SIZE
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = set()
        self.reset()
    
    def reset(self):
        # start and goal positions
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_width - 1, self.grid_height - 1)
        self.create_obstacles()
        return self.agent_pos

    def create_obstacles(self):
        self.obstacles = set()
        #obstacles added
        for i in range(5, 15):
            self.obstacles.add((i, 10))
        for i in range(2, 8):
            self.obstacles.add((10, i))
    
    def valid_position(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
        if pos in self.obstacles:
            return False
        return True

    def step(self, action):
        # Moves: up, down, left, right
        x, y = self.agent_pos
        if action == 'UP':
            new_pos = (x, y - 1)
        elif action == 'DOWN':
            new_pos = (x, y + 1)
        elif action == 'LEFT':
            new_pos = (x - 1, y)
        elif action == 'RIGHT':
            new_pos = (x + 1, y)
        else:
            new_pos = (x, y)

        # Check for collisions: wall or obstacle
        if not self.valid_position(new_pos):
            # Collision penalty
            reward = -10
            done = True
            return self.agent_pos, reward, done

        self.agent_pos = new_pos

        # Check if reached goal:
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
        else:
            reward = -1  # Step cost
            done = False
        
        return self.agent_pos, reward, done

    def get_state(self):
        x, y = self.agent_pos
        gx, gy = self.goal_pos
        dx = gx - x
        dy = gy - y
        distance = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        return (x, y), distance, angle

# Q-Learning Agent Class
class QAgent:
    def __init__(self, env):
        self.env = env
        # every grid cell is a state.
        self.q_table = {}
        for i in range(env.grid_width):
            for j in range(env.grid_height):
                self.q_table[(i, j)] = {action: 0 for action in ACTIONS}

    def choose_action(self, state, epsilon):
        # Epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            return random.choice(ACTIONS)
        else:
            state_actions = self.q_table[state]
            max_val = max(state_actions.values())
            best_actions = [action for action, value in state_actions.items() if value == max_val]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        # Q-learning update equation
        self.q_table[state][action] = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)

# Visualization Functions

def draw_grid(screen, env):
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WINDOW_WIDTH, y))
    
    # Draw obstacles
    for (i, j) in env.obstacles:
        rect = pygame.Rect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLACK, rect)
    
    # Draw goal
    gx, gy = env.goal_pos
    goal_rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, GREEN, goal_rect)

def draw_agent(screen, pos):
    x, y = pos
    center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
    pygame.draw.circle(screen, RED, center, CELL_SIZE // 3)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("2D Autonomous Agent with Q-Learning")
    clock = pygame.time.Clock()

    env = GridWorld()
    agent = QAgent(env)
    
    # Logging
    episode_performance = []
    
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        steps = 0
        collisions = 0
        done = False

        while not done and steps < MAX_STEPS:
           
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.choose_action(state, EPSILON)
            next_state, reward, done = env.step(action)
            if reward == -10:
                collisions += 1
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1

            # Visualization 
            screen.fill(WHITE)
            draw_grid(screen, env)
            draw_agent(screen, state)
            pygame.display.flip()
            clock.tick(FPS)
        
        episode_performance.append({
            'episode': episode,
            'steps': steps,
            'total_reward': total_reward,
            'collisions': collisions
        })
        # Print progress every 50 episodes
        if episode % 50 == 0:
            print(f"Episode {episode} - Steps: {steps} Reward: {total_reward} Collisions: {collisions}")
    
    state = env.reset()
    done = False
    demo_steps = 0
    while not done and demo_steps < MAX_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        action = agent.choose_action(state, epsilon=0) 
        next_state, reward, done = env.step(action)
        state = next_state
        demo_steps += 1
        
        screen.fill(WHITE)
        draw_grid(screen, env)
        draw_agent(screen, state)
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    
    # Log final performance summary
    print("Training complete.")
    avg_steps = np.mean([p['steps'] for p in episode_performance])
    print(f"Average steps per episode: {avg_steps:.2f}")

if __name__ == "__main__":
    main()
