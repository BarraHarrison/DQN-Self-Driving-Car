import pygame
import sys
import torch
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
LOAD_MODEL = True
EVAL_ONLY = True
MODEL_PATH = "checkpoints/dqn_episode_50_reward_412.pth"

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Driving Car - DQN")
    pygame.event.pump()
    clock = pygame.time.Clock()

    original_map = pygame.image.load("assets/new_map.png").convert()
    track = pygame.transform.scale(original_map, (WIDTH, HEIGHT))
    env = Environment(track)

    state_size = 5
    action_size = 3

    agent = DQNAgent(state_size, action_size)
    memory = ReplayBuffer(REPLAY_CAPACITY)

    if LOAD_MODEL:
            agent.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            agent.model.eval()
            agent.epsilon = 0.1
            print(f"📥 Loaded model from: {MODEL_PATH}")


    spawn_x, spawn_y = 460, 600
    START_LINE_RECT = pygame.Rect(440, 590, 40, 20)
    print("Pixel color at spawn:", original_map.get_at((spawn_x, spawn_y)))
    episode_rewards = []

    for episode in range(MAX_EPISODES):
        car = Car(spawn_x, spawn_y)
        lap_count = 0
        was_far_from_start = False
        car.angle = 0
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
            car_pos = pygame.Vector2(car.x, car.y)

            if not START_LINE_RECT.collidepoint(car_pos):
                was_far_from_start = True

            if was_far_from_start and START_LINE_RECT.collidepoint(car_pos):
                lap_count += 1
                was_far_from_start = False
                print(f"🏁 Lap completed! Total laps: {lap_count}")

            car.draw(screen)

            next_state = env.get_sensor_distances(car)
            reward, done = env.calculate_reward(car)
            total_reward += reward

            if not EVAL_ONLY:
                memory.push(state, action, reward, next_state, done)

            if not EVAL_ONLY and len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                for s, a, r, s_next, d in zip(*batch):
                    agent.train_step(s, a, r, s_next, d)

            pygame.draw.circle(screen, (0, 255, 0), (int(car.x), int(car.y)), 5)
            pygame.display.flip()
            clock.tick(FPS)

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f} | Laps: {lap_count} | Epsilon: {agent.epsilon:.3f}")

        if not EVAL_ONLY and total_reward >= 400:
            model_path = f"checkpoints/dqn_episode_{episode + 1}_reward_{int(total_reward)}.pth"
            torch.save(agent.model.state_dict(), model_path)
            print(f"✅ Model saved: {model_path}")

        if not EVAL_ONLY and (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()


    pygame.quit()
    sys.exit()

    if not EVAL_ONLY:
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress")
        plt.show()

if __name__ == "__main__":
    main()