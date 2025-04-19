import pygame
import time
import sys
import torch
import math
import matplotlib.pyplot as plt
import pygame.freetype
from car import Car
from environment import Environment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer


WIDTH, HEIGHT = 800, 700
FPS = 60
MAX_EPISODES = 500
BATCH_SIZE = 64
REPLAY_CAPACITY = 10000
TARGET_UPDATE_FREQ = 10
LOAD_MODEL = True
EVAL_ONLY = True
MODEL_PATH = "checkpoints/dqn_episode_50_reward_412.pth"
LAP_COMPLETION_RADIUS = 100

def main():
    pygame.init()
    pygame.freetype.init()
    font = pygame.freetype.SysFont("Arial", 24)
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
    START_LINE_RECT = pygame.Rect(430, 580, 80, 40)
    pygame.draw.rect(screen, (0, 255, 255), START_LINE_RECT, 2)
    print("Pixel color at spawn:", original_map.get_at((spawn_x, spawn_y)))
    episode_rewards = []

    for episode in range(MAX_EPISODES):
        car = Car(spawn_x, spawn_y)
        lap_count = 0
        was_far_enough = False
        lap_cooldown = 0
        start_pos = pygame.Vector2(spawn_x, spawn_y)
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
            distance_to_start = math.hypot(car.x - spawn_x, car.y - spawn_y)
            # print(f"Distance to start: {distance_to_start:.2f}, Was far enough: {was_far_enough}, Lap cooldown: {lap_cooldown}")

            if distance_to_start > 150:
                was_far_enough = True

            if was_far_enough and distance_to_start <= LAP_COMPLETION_RADIUS and lap_cooldown == 0:
                car.lap_count += 1
                lap_end_time = time.time()
                lap_duration = lap_end_time - car.lap_start_time
                car.lap_times.append(lap_duration)
                print(f"🏁 Lap completed! Total laps: {lap_count}")
                print(f"⏱️ Lap time: {lap_duration:.2f} seconds")
                car.lap_start_time = lap_end_time
                was_far_enough = False
                lap_cooldown = 60


            if lap_cooldown > 0:
                lap_cooldown -= 1

            next_state = env.get_sensor_distances(car)

            reward, done = env.calculate_reward(car)
            total_reward += reward

            car.draw(screen)
            pygame.draw.circle(screen, (0, 255, 0), (int(car.x), int(car.y)), 5)

            font.render_to(screen, (10, 10), f"Laps: {car.lap_count}", (0, 0, 0))
            if car.lap_times:
                last_lap_time = car.lap_times[-1]
                font.render_to(screen, (10, 40), f"Last Lap: {last_lap_time:.2f}s", (0, 0, 0))

            pygame.display.flip()
            clock.tick(FPS)

            if not EVAL_ONLY:
                memory.push(state, action, reward, next_state, done)

            if not EVAL_ONLY and len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                for s, a, r, s_next, d in zip(*batch):
                    agent.train_step(s, a, r, s_next, d)

            pygame.draw.circle(screen, (0, 255, 0), (int(car.x), int(car.y)), 5)
            clock.tick(FPS)
            pygame.display.flip()

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