import pygame
import math

class Environment:
    def __init__(self, track_surface):
        self.track_surface = track_surface

    def get_sensor_distances(self, car):
        distances = []
        for angle_offset in car.sensor_angles:
            sensor_angle = math.radians(car.angle + angle_offset)
            for dist in range(car.sensor_length):
                sensor_x = int(car.x + dist * math.cos(sensor_angle))
                sensor_y = int(car.y - dist * math.sin(sensor_angle))

                if 0 <= sensor_x < self.track_surface.get_width() and 0 <= sensor_y < self.track_surface.get_height():
                    color = self.track_surface.get_at((sensor_x, sensor_y))
                    if color != pygame.Color(0, 0, 0, 255):
                        break
                else:
                    break

            distances.append(dist / car.sensor_length)

        return distances

    def check_collision(self, car):
        car_mask = pygame.mask.from_surface(car.image)
        track_mask = pygame.mask.from_threshold(self.track_surface, (0, 0, 0, 255), (1, 1, 1, 255))

        offset = (int(car.rect.left), int(car.rect.top))
        overlap = track_mask.overlap(car_mask, offset)
        return overlap is None

    def calculate_reward(self, car):
        if self.check_collision(car):
            return -10, True
        return +1, False
