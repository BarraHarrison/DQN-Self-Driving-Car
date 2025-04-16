import pygame
import sys
from car import Car
from environment import Environment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


WIDTH, HEIGHT = 800, 700
FPS = 60
MAX_EPISODES = 500
BATCH_SIZE = 64
REPLAY_CAPACITY = 10000
TARGET_UPDATE_FREQ = 10

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Driving Car - DQN")
    pygame.event.pump()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    original_map = pygame.image.load("assets/new_map.png").convert()
    track = pygame.transform.scale(original_map, (WIDTH, HEIGHT))
    env = Environment(track)

    state_size = 5
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    memory = ReplayBuffer(REPLAY_CAPACITY)

    episode_rewards = []

    for episode in range(MAX_EPISODES):
        car = Car(400, 660)
        spawn_attempts = 0
        while env.check_collision(car):
            car = Car(car.x, car.y - 1)
            spawn_attempts += 1
            if spawn_attempts > 30:
                print("Failed to spawn on road. Skipping episode.")
                continue
            
        total_reward = 0
        done = False

        while not done:
            screen.blit(track, (0, 0))

            state = env.get_sensor_distances(car)

            action = agent.act(state)

            if action == 1:
                car.rotate(5)
            elif action == 2:
                car.rotate(-5)
            car.move()
            car.update_sensors(track)
            car.draw(screen)

            next_state = env.get_sensor_distances(car)
            reward, done = env.calculate_reward(car)
            total_reward += reward

            memory.push(state, action, reward, next_state, done)

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                for s, a, r, s_next, d in zip(*batch):
                    agent.train_step(s, a, r, s_next, d)

            pygame.draw.circle(screen, (0, 255, 0), (int(car.x), int(car.y)), 5)
            pygame.display.flip()
            clock.tick(FPS)

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

    pygame.quit()
    sys.exit()

    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

if __name__ == "__main__":
    main()