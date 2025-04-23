import pygame
import numpy as np
import random
import sys

# Environment Settings
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 30
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10

# Q-Learning parameters
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.3       # exploration rate
EPISODES = 1000     # number of training episodes
MAX_STEPS = 200     # max steps per episode

# RGB Colors
WHITE = (255, 255, 255)
GRAY  = (180, 180, 180)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)

class GridWorld:
    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.cell_size = CELL_SIZE
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_width - 1, self.grid_height - 1)
        self._make_obstacles()
        return self.agent_pos

    def _make_obstacles(self):
        self.obstacles = set()
        for i in range(5, 15):
            self.obstacles.add((i, 10))
        for j in range(2, 8):
            self.obstacles.add((10, j))

    def valid(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
        return pos not in self.obstacles

    def step(self, action):
        x, y = self.agent_pos
        if action == 'UP':    new = (x, y - 1)
        elif action == 'DOWN':new = (x, y + 1)
        elif action == 'LEFT':new = (x - 1, y)
        elif action == 'RIGHT':new = (x + 1, y)
        else:                 new = (x, y)

        if not self.valid(new):
            # collision penalty, continue
            return self.agent_pos, -10, False

        self.agent_pos = new
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 100, True
        return self.agent_pos, -1, False

class QAgent:
    def __init__(self, env):
        self.env = env
        self.n_actions = len(ACTIONS)
        self.q_table = np.zeros((env.grid_width,
                                 env.grid_height,
                                 self.n_actions))
        self.action_index = {act: i for i, act in enumerate(ACTIONS)}

    def choose_action(self, state, epsilon):
        x, y = state
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        q_vals = self.q_table[x, y]
        best_idx = int(np.argmax(q_vals))
        return ACTIONS[best_idx]

    def learn(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        a_idx = self.action_index[action]
        td_target = reward + DISCOUNT_FACTOR * self.q_table[nx, ny].max()
        self.q_table[x, y, a_idx] += LEARNING_RATE * (td_target - self.q_table[x, y, a_idx])

# Visualization Helpers

def draw_grid(screen, env):
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WINDOW_WIDTH, y))
    for ox, oy in env.obstacles:
        rect = pygame.Rect(ox*CELL_SIZE, oy*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLACK, rect)
    gx, gy = env.goal_pos
    pygame.draw.rect(screen, GREEN, (gx*CELL_SIZE, gy*CELL_SIZE, CELL_SIZE, CELL_SIZE))


def draw_agent(screen, pos):
    x, y = pos
    center = (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2)
    pygame.draw.circle(screen, RED, center, CELL_SIZE//3)


def train_agent(env, agent):
    for ep in range(EPISODES):
        state = env.reset()
        for _ in range(MAX_STEPS):
            action = agent.choose_action(state, EPSILON)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            if done:
                break
        if ep % 100 == 0:
            print(f"Episode {ep}/{EPISODES} completed")
    print("Training complete.")


def demo_agent(env, agent):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Q-Learning Demo")
    clock = pygame.time.Clock()

    state = env.reset()
    done = False
    demo_steps = 0
    visited = {}
    while not done and demo_steps < MAX_STEPS * 2:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = agent.choose_action(state, epsilon=0)
        next_state, _, done = env.step(action)

        # loop detection
        visited[next_state] = visited.get(next_state, 0) + 1
        if visited[next_state] > 4:
            print(f"Stuck in loop at {next_state}, breaking demo.")
            break

        state = next_state
        demo_steps += 1

        screen.fill(WHITE)
        draw_grid(screen, env)
        draw_agent(screen, state)
        pygame.display.flip()
        clock.tick(FPS)

    if not done:
        print("Demo ended without reaching the goal.")
    pygame.quit()


if __name__ == "__main__":
    env = GridWorld()
    agent = QAgent(env)
    train_agent(env, agent)
    demo_agent(env, agent)
