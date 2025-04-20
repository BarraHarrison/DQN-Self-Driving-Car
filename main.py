import pygame
import time
import sys
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import imageio
import pygame.freetype
import os
import csv
import io
from io import BytesIO
from PIL import Image
from car import Car
from environment import Environment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


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

def export_lap_times(lap_times, filename="lap_times.csv"):
    start_lap_number = 1
    if os.path.exists(filename):
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            next(reader, None)
            existing_laps = list(reader)
            if existing_laps:
                start_lap_number = int(existing_laps[-1][0]) + 1

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if os.stat(filename).st_size == 0:
            writer.writerow(["Lap Number", "Lap Time (seconds)"])
        for i, laptime in enumerate(lap_times, start=start_lap_number):
            writer.writerow([i, round(laptime, 2)])
    print(f"📤 Lap times appended to {filename}")

def render_reward_plot(rewards, width=300, height=200):
    plt.figure(figsize=(3, 2))
    plt.plot(rewards, color='blue')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    img = Image.open(buf).convert("RGB")
    img = img.resize((width, height))

    mode = img.mode
    size = img.size
    data = img.tobytes()
    reward_surface = pygame.image.fromstring(data, size, mode)

    return reward_surface

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
    all_lap_times = []
    frames = []

    for episode in range(MAX_EPISODES):
        car = Car(spawn_x, spawn_y)
        car.lap_times = []
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

            if distance_to_start > 150:
                was_far_enough = True

            if was_far_enough and distance_to_start <= LAP_COMPLETION_RADIUS and lap_cooldown == 0:
                lap_count += 1
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

            if "reward_plot_surface" in locals():
                plot_x = (WIDTH - reward_plot_surface.get_width()) // 2
                plot_y = (HEIGHT - reward_plot_surface.get_height()) // 2
                screen.blit(reward_plot_surface, (plot_x, plot_y))

            pygame.draw.circle(screen, (0, 255, 0), (int(car.x), int(car.y)), 5)

            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)
            pygame.display.flip()
            clock.tick(FPS)

        episode_rewards.append(total_reward)

        if episode % 3 == 0:
            reward_plot_surface = render_reward_plot(episode_rewards)

        all_lap_times.extend(car.lap_times)
        print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f} | Laps: {lap_count} | Epsilon: {agent.epsilon:.3f}")

        if not EVAL_ONLY and total_reward >= 400:
            model_path = f"checkpoints/dqn_episode_{episode + 1}_reward_{int(total_reward)}.pth"
            torch.save(agent.model.state_dict(), model_path)
            print(f"✅ Model saved: {model_path}")

        if not EVAL_ONLY and (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        if car.lap_times:
            all_lap_times.extend(car.lap_times)
            export_lap_times(all_lap_times)


    if frames:
        imageio.mimsave("simulation_replay.gif", frames, fps=30)
        print("🎥 Simulation replay saved as simulation_replay.gif")
    
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