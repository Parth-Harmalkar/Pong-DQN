import pygame
import os
import sys
import gc
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque

# GPU memory setting
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Define Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BALL_SIZE = 30
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 140
BALL_SPEED = 9
PADDLE_SPEED = 10
MAX_EPISODES = 10000
MAX_STEPS = 4000
BATCH_SIZE = 300
GAMMA = 0.99
EPSILON = 1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
LEARNING_RATE = 0.001
MEMORY_SIZE = 50000
TARGET_UPDATE_FREQ = 100


# Initialize pygame
pygame.init()
clock = pygame.time.Clock()

# Setup up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("PONG with DQN")

# Define Colors
BG_COLOR = pygame.Color("grey12")
LIGHT_GREY = (200, 200, 200)

# Create a font object 
game_font = pygame.font.Font("freesansbold.ttf", 28)

# Define the DQN model
def create_dqn_model():
    inputs = tf.keras.Input(shape=(12,))
    layer = tf.keras.layers.Dense(128, activation="relu")(inputs)
    layer = tf.keras.layers.Dense(128, activation="relu")(layer)
    layer = tf.keras.layers.Dense(64, activation="relu")(layer)
    action = tf.keras.layers.Dense(3, activation="linear")(layer)

    return tf.keras.Model(inputs=inputs, outputs=action)

# Define Pong environment
class PongEnv:
    def __init__(self):
        self.ball = pygame.Rect(SCREEN_WIDTH / 2 - BALL_SIZE / 2, SCREEN_HEIGHT / 2 - BALL_SIZE / 2, BALL_SIZE, BALL_SIZE)
        self.player = pygame.Rect(SCREEN_WIDTH - PADDLE_WIDTH - 10, SCREEN_HEIGHT / 2 - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.opponent = pygame.Rect(15, SCREEN_HEIGHT / 2 - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball_speed_x = BALL_SPEED * random.choice((-1,1))
        self.ball_speed_y = BALL_SPEED * random.choice((-1,1))
        self.player_speed = 0
        self.opponent_speed = PADDLE_SPEED
        self.player_score = 0
        self.opponent_score = 0
        self.player_hits = 0
        self.speed_increased = True
        self.dist_ball_player = math.sqrt((self.ball.centerx - self.player.centerx)**2 + (self.ball.centery - self.player.centery)**2)
        self.dist_ball_opponent = math.sqrt((self.ball.centerx - self.opponent.centerx)**2 + (self.ball.centery - self.opponent.centery)**2)


    def reset(self):
        self.ball.center =  (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        self.player.center = (SCREEN_WIDTH - PADDLE_WIDTH / 2 - 10, SCREEN_HEIGHT / 2)
        self.opponent.center = (15, SCREEN_HEIGHT / 2)
        self.ball_speed_x = BALL_SPEED * random.choice((-1,1))
        self.ball_speed_y = BALL_SPEED * random.choice((-1,1))
        self.player_speed = 0
        self.opponent_speed = PADDLE_SPEED
        self.player_hits = 0
        self.dist_ball_player = math.sqrt((self.ball.centerx - self.player.centerx)**2 + (self.ball.centery - self.player.centery)**2)
        self.dist_ball_opponent = math.sqrt((self.ball.centerx - self.opponent.centerx)**2 + (self.ball.centery - self.opponent.centery)**2)

        return self.get_state()
    
    def get_state(self):
        state = [
            self.player_speed / PADDLE_SPEED,
            self.player.centerx / SCREEN_WIDTH,
            self.player.centery / SCREEN_HEIGHT,
            self.opponent_speed / PADDLE_SPEED,
            self.opponent.centerx / SCREEN_WIDTH,
            self.opponent.centery / SCREEN_HEIGHT,
            self.ball_speed_x / BALL_SPEED,
            self.ball_speed_y / BALL_SPEED,
            self.ball.centerx / SCREEN_WIDTH,
            self.ball.centery / SCREEN_HEIGHT,
            self.dist_ball_player / (math.sqrt(SCREEN_HEIGHT**2 + SCREEN_WIDTH**2)),
            self.dist_ball_opponent / (math.sqrt(SCREEN_HEIGHT**2 + SCREEN_WIDTH**2))
        ]

        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        reward = 0
        done = False
               

        # Player action
        if action == 1:
            # UP
            self.player_speed = -PADDLE_SPEED
        elif action == 2:
            # DOWN
            self.player_speed = PADDLE_SPEED
        else:
            self.player_speed = 0

        # Move player
        self.player.centery += self.player_speed

        if self.player.top <= 0:
            self.player.top = 0
        if self.player.bottom >= SCREEN_HEIGHT:
            self.player.bottom = SCREEN_HEIGHT

        # Move opponent
        if self.opponent.centery < self.ball.centery:
            self.opponent.y += self.opponent_speed
        if self.opponent.centery > self.ball.centery:
            self.opponent.y -= self.opponent_speed
        
        if self.opponent.top <= 0:
            self.opponent.top = 0
        if self.opponent.bottom >= SCREEN_HEIGHT:
            self.opponent.bottom = SCREEN_HEIGHT

        # Increasing ball speed after certain hits
        if self.player_hits in [3, 6, 9, 12, 15] and not self.speed_increased:
            self.ball_speed_x -= 1
            self.ball_speed_y -= 1
            self.speed_increased = True
        elif self.player_hits not in [3, 6, 9, 12, 15]:
            self.speed_increased = False

        # Move ball
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        # Ball collision with walls
        if self.ball.top <= 0 or self.ball.bottom >= SCREEN_HEIGHT:
            self.ball_speed_y *= -1
        
        self.dist_ball_player = math.sqrt((self.ball.centerx - self.player.centerx)**2 + (self.ball.centery - self.player.centery)**2)
        self.dist_ball_opponent = math.sqrt((self.ball.centerx - self.opponent.centerx)**2 + (self.ball.centery - self.opponent.centery)**2)

        # Reward for being closer to the ball
        if self.ball.centerx > SCREEN_WIDTH / 2 + 100  and self.ball_speed_x > 0:
            if self.dist_ball_player <= 400:
                reward += 0.1 * (400 - self.dist_ball_player) / (400)
            else:
                reward -= 0.1 * (400 - self.dist_ball_player) / (400)

        # Ball collision with paddles
        if self.ball.colliderect(self.player) and self.ball_speed_x > 0:
            self.player_hits += 1
            # print(self.player_hits)

            # Reward for hitting the ball consecutively
            if self.player_hits <= 3:
                reward += 2.0
            elif 3 < self.player_hits <= 6:
                reward += 10
            elif self.player_hits > 6:
                reward += 10
                
            if abs(self.ball.right - self.player.left) < 10:
                self.ball_speed_x *= -1
            elif abs(self.ball.bottom - self.player.top) < 10 and self.ball_speed_y > 0:
                self.ball_speed_y *= -1
            elif abs(self.ball.top - self.player.bottom) < 10 and self.ball_speed_y < 0:
                self.ball_speed_y *= -1

        if self.ball.colliderect(self.opponent) and self.ball_speed_x < 0:
            # reward -= 1
            if abs(self.ball.left - self.opponent.right) < 10:
                self.ball_speed_x *= -1               
            elif abs(self.ball.bottom - self.opponent.top) < 10 and self.ball_speed_y > 0:
                self.ball_speed_y *= -1
            elif abs(self.ball.top - self.opponent.bottom) < 10 and self.ball_speed_y < 0:
                self.ball_speed_y *= -1

        # Reward for playing the game every frame
        reward += 0.01
        
        # Ball out of bounds
        if self.ball.left <= 0:
            self.player_score += 1
            reward += 50
            done = True
        if self.ball.right >= SCREEN_WIDTH:
            self.opponent_score += 1
            reward -= 50
            done = True
        
        return self.get_state(), reward, done
    
# Checkpoint directory
checkpoint_dir = './checkpoints'
graph_dir = './graphs'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
    
# Initialize DQN
model = create_dqn_model()
target_model = create_dqn_model()
target_model.set_weights(model.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_function = tf.keras.losses.Huber()

model.compile(optimizer=optimizer, loss=loss_function)

# Experience replay buffer
memory = deque(maxlen=MEMORY_SIZE)

# Initialize the environment
env = PongEnv()

# Initialize array for plotting 
reward_store = [0]


# Training loop
for episode in range(MAX_EPISODES):
    tf.keras.backend.clear_session()
    state = env.reset()
    state = np.expand_dims(state, axis=0)

    total_reward = 0
    
    for step in range(MAX_STEPS):
        # Handle Pygame events to keep the game responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Epsion Greedy action selection
        if np.random.rand() < EPSILON:
            action = np.random.choice(3)    
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            # print(action)
        
        # Take action
        next_state, reward, done = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        total_reward += reward

        # Display the game
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, LIGHT_GREY, env.player)
        pygame.draw.rect(screen, LIGHT_GREY, env.opponent)
        pygame.draw.ellipse(screen, LIGHT_GREY, env.ball)
        pygame.draw.aaline(screen, LIGHT_GREY, (SCREEN_WIDTH / 2, 0), (SCREEN_WIDTH / 2, SCREEN_HEIGHT))

        # Score text
        player_text = game_font.render(f"{env.player_score}", False, LIGHT_GREY)
        screen.blit(player_text, (SCREEN_WIDTH / 2 + 150, SCREEN_HEIGHT / 2))
        opponent_text = game_font.render(f"{env.opponent_score}", False, LIGHT_GREY)
        screen.blit(opponent_text, (SCREEN_WIDTH / 2 - 200, SCREEN_HEIGHT / 2))
    
        # Update the display
        pygame.display.flip()
        clock.tick(60)

        # Store experience in memory buffer
        memory.append((state, action, reward, next_state, done))

        # Update state
        state = next_state

        # Check if done
        if done:
            reward_store.append(total_reward)
            break
        
        # Experience Replay
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = np.concatenate(states)
            next_states = np.concatenate(next_states)

            target_q_values = target_model.predict(next_states, verbose=0)
            max_target_q_values = np.amax(target_q_values, axis=1)

            targets = model.predict(states, verbose=0)
            for i in range(BATCH_SIZE):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + GAMMA * max_target_q_values[i]
            
            model.train_on_batch(states, targets)

        # Update target model
        if step % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())
        
        # After each step garbage collection
        gc.collect()

    
    # Reduce Episolon
    if EPSILON >= EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    # Saving model information
    if episode % 100 == 0 and episode != 0: 
        model.save(os.path.join(checkpoint_dir, f'model_episode_{episode}.h5'))
        avg_reward = np.mean(reward_store[-100])
        print(f"Average_reward : {avg_reward:.2f}")

        # Plotting graph
        x = np.arange(0, len(reward_store), 1)
        plt.figure(figsize=(10,6))
        plt.plot(x, np.array(reward_store), label="Reward Graph")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.legend()
        plt.savefig(os.path.join(graph_dir, f'reward_graph_episode{episode}.png'))
        plt.close()

    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {EPSILON}")
    print(f"Total Hits : {env.player_hits}")


    # After each episode garbage collection
    gc.collect()

print(f"Total Games: {env.player_score + env.opponent_score}")
print(f"Games won : {env.player_score}")

# Save the model
model.save("Pong_dqn_model_01.h5")



# # Setup up the display
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("PONG with DQN")


# # Play the game using the trained model
# env = PongEnv()
# state = env.reset()
# state = np.expand_dims(state, axis=0)

# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             exit()

#     q_values = model.predict(state, verbose=0)
#     action = np.argmax(q_values[0])
#     next_state, reward, done = env.step(action)
#     next_state = np.expand_dims(next_state, axis=0)

#     state = next_state

#     if done:
#         state = env.reset()
#         state = np.expand_dims(state, axis=0)

#     # Display the game
#     screen.fill(BG_COLOR)
#     pygame.draw.rect(screen, LIGHT_GREY, env.player)
#     pygame.draw.rect(screen, LIGHT_GREY, env.opponent)
#     pygame.draw.ellipse(screen, LIGHT_GREY, env.ball)
#     pygame.draw.aaline(screen, LIGHT_GREY, (SCREEN_WIDTH / 2, 0), (SCREEN_WIDTH / 2, SCREEN_HEIGHT))

#     # Score text
#     player_text = game_font.render(f"{env.player_score}", False, LIGHT_GREY)
#     screen.blit(player_text, (SCREEN_WIDTH / 2 + 20, SCREEN_HEIGHT / 2))
#     opponent_text = game_font.render(f"{env.opponent_score}", False, LIGHT_GREY)
#     screen.blit(opponent_text, (SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2))
    
#     # Update the display
#     pygame.display.flip()
#     clock.tick(60)
