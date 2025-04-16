import pygame
import math

class Car:
    def __init__(self, x, y):
        car_image = pygame.image.load("assets/car.png").convert_alpha()
        scaled_size = (40, 20)
        self.original_image = pygame.transform.scale(car_image, scaled_size)

        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 5

        self.image = self.original_image
        self.update_image()

        self.sensors = []
        self.sensor_length = 100
        self.sensor_angles = [-90, -45, 0, 45, 90]

    def update_image(self):
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def move(self):
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)
        self.rect.center = (self.x, self.y)

    def rotate(self, direction):
        self.angle += direction
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def update_sensors(self, track_surface):
        self.sensors = []
        for angle_offset in self.sensor_angles:
            sensor_angle = math.radians(self.angle + angle_offset)
            for dist in range(self.sensor_length):
                sensor_x = int(self.x + dist * math.cos(sensor_angle))
                sensor_y = int(self.y - dist * math.sin(sensor_angle))

                if 0 <= sensor_x < track_surface.get_width() and 0 <= sensor_y < track_surface.get_height():
                    if track_surface.get_at((sensor_x, sensor_y)) == pygame.Color(0, 0, 0, 255):
                        break
                self.sensors.append((sensor_x, sensor_y))

    def draw(self, screen):
        screen.blit(self.image, self.rect)

        for point in self.sensors:
            pygame.draw.circle(screen, (255, 0, 0), point, 2)
