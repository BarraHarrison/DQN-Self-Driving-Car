import pygame
import sys
from car import Car
from environment import Environment

WIDTH, HEIGHT = 800, 700
FPS = 60

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Driving Car - DQN")
    clock = pygame.time.Clock()

    track = pygame.image.load("assets/new_map.png").convert()
    car = Car(420, 640)
    env = Environment(track)

    running = True
    while running:
        screen.blit(track, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            car.rotate(5)
        if keys[pygame.K_RIGHT]:
            car.rotate(-5)

        car.move()
        car.update_sensors(track)
        car.draw(screen)

        sensor_data = env.get_sensor_distances(car)
        reward, done = env.calculate_reward(car)

        print(f"Sensor: {sensor_data} | Reward: {reward} | Done: {done}")

        if done:
            print("Collision! Resetting car...")
            car = Car(420, 420)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
