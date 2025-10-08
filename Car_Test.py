import pygame
import numpy as np
import math
import random
import sys

# Inicialização do Pygame
pygame.init()

# Configurações da tela
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Carrinho Autônomo com IA")

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# Parâmetros da pista
TRACK_WIDTH = 300
TRACK_HEIGHT = 400
TRACK_MARGIN = 100

# Parâmetros do carrinho
CAR_WIDTH = 20
CAR_HEIGHT = 30
MAX_SPEED = 5
ACCELERATION = 0.1
ROTATION_SPEED = 3

# Parâmetros da IA
NUM_SENSORS = 5
SENSOR_RANGE = 200
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.angle = 0  # Ângulo em graus
        self.speed = 0
        self.sensors = [0] * NUM_SENSORS
        self.alive = True
        self.score = 0
        self.checkpoints = 0
        self.total_time = 0
        
    def update(self, action):
        if not self.alive:
            return
            
        # Aplicar ação
        if action == 0:  # Acelerar
            self.speed = min(self.speed + ACCELERATION, MAX_SPEED)
        elif action == 1:  # Frear
            self.speed = max(self.speed - ACCELERATION, 0)
        elif action == 2:  # Virar à esquerda
            self.angle -= ROTATION_SPEED
        elif action == 3:  # Virar à direita
            self.angle += ROTATION_SPEED
            
        # Movimentar o carro
        rad_angle = math.radians(self.angle)
        self.x += self.speed * math.sin(rad_angle)
        self.y -= self.speed * math.cos(rad_angle)
        
        # Atualizar sensores
        self.update_sensors()
        
        # Verificar colisões
        self.check_collision()
        
        # Atualizar tempo
        self.total_time += 1
        
    def update_sensors(self):
        for i in range(NUM_SENSORS):
            sensor_angle = self.angle - 90 + (i * 180 / (NUM_SENSORS - 1))
            rad_angle = math.radians(sensor_angle)
            
            # Calcular posição final do sensor
            end_x = self.x + SENSOR_RANGE * math.sin(rad_angle)
            end_y = self.y - SENSOR_RANGE * math.cos(rad_angle)
            
            # Verificar interseção com as bordas da pista
            distance = self.cast_ray(self.x, self.y, end_x, end_y)
            self.sensors[i] = distance / SENSOR_RANGE  # Normalizar
            
    def cast_ray(self, start_x, start_y, end_x, end_y):
        # Verificar interseção com as bordas da pista
        track_left = WIDTH // 2 - TRACK_WIDTH // 2
        track_right = WIDTH // 2 + TRACK_WIDTH // 2
        track_top = HEIGHT // 2 - TRACK_HEIGHT // 2
        track_bottom = HEIGHT // 2 + TRACK_HEIGHT // 2
        
        # Verificar interseção com cada borda
        distances = []
        
        # Borda esquerda
        if start_x != end_x:
            t = (track_left - start_x) / (end_x - start_x)
            if 0 <= t <= 1:
                y = start_y + t * (end_y - start_y)
                if track_top <= y <= track_bottom:
                    distance = math.sqrt((track_left - start_x)**2 + (y - start_y)**2)
                    distances.append(distance)
        
        # Borda direita
        if start_x != end_x:
            t = (track_right - start_x) / (end_x - start_x)
            if 0 <= t <= 1:
                y = start_y + t * (end_y - start_y)
                if track_top <= y <= track_bottom:
                    distance = math.sqrt((track_right - start_x)**2 + (y - start_y)**2)
                    distances.append(distance)
        
        # Borda superior
        if start_y != end_y:
            t = (track_top - start_y) / (end_y - start_y)
            if 0 <= t <= 1:
                x = start_x + t * (end_x - start_x)
                if track_left <= x <= track_right:
                    distance = math.sqrt((x - start_x)**2 + (track_top - start_y)**2)
                    distances.append(distance)
        
        # Borda inferior
        if start_y != end_y:
            t = (track_bottom - start_y) / (end_y - start_y)
            if 0 <= t <= 1:
                x = start_x + t * (end_x - start_x)
                if track_left <= x <= track_right:
                    distance = math.sqrt((x - start_x)**2 + (track_bottom - start_y)**2)
                    distances.append(distance)
        
        return min(distances) if distances else SENSOR_RANGE
    
    def check_collision(self):
        # Verificar se o carro está fora da pista
        track_left = WIDTH // 2 - TRACK_WIDTH // 2
        track_right = WIDTH // 2 + TRACK_WIDTH // 2
        track_top = HEIGHT // 2 - TRACK_HEIGHT // 2
        track_bottom = HEIGHT // 2 + TRACK_HEIGHT // 2
        
        # Pontos do carro (considerando rotação)
        points = []
        rad_angle = math.radians(self.angle)
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)
        
        # Calcular os quatro cantos do carro
        half_width = self.width / 2
        half_height = self.height / 2
        
        points.append((
            self.x + half_width * cos_a - half_height * sin_a,
            self.y + half_width * sin_a + half_height * cos_a
        ))
        points.append((
            self.x - half_width * cos_a - half_height * sin_a,
            self.y - half_width * sin_a + half_height * cos_a
        ))
        points.append((
            self.x - half_width * cos_a + half_height * sin_a,
            self.y - half_width * sin_a - half_height * cos_a
        ))
        points.append((
            self.x + half_width * cos_a + half_height * sin_a,
            self.y + half_width * sin_a - half_height * cos_a
        ))
        
        # Verificar se algum ponto está fora da pista
        for x, y in points:
            if x < track_left or x > track_right or y < track_top or y > track_bottom:
                self.alive = False
                self.score -= 50  # Penalidade por sair da pista
                return
    
    def get_state(self):
        # Retorna o estado atual (sensores + velocidade)
        return self.sensors + [self.speed / MAX_SPEED]
    
    def draw(self, screen):
        if not self.alive:
            return
            
        # Desenhar carro
        rad_angle = math.radians(self.angle)
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)
        
        # Calcular os quatro cantos do carro
        half_width = self.width / 2
        half_height = self.height / 2
        
        points = [
            (self.x + half_width * cos_a - half_height * sin_a,
             self.y + half_width * sin_a + half_height * cos_a),
            (self.x - half_width * cos_a - half_height * sin_a,
             self.y - half_width * sin_a + half_height * cos_a),
            (self.x - half_width * cos_a + half_height * sin_a,
             self.y - half_width * sin_a - half_height * cos_a),
            (self.x + half_width * cos_a + half_height * sin_a,
             self.y + half_width * sin_a - half_height * cos_a)
        ]
        
        pygame.draw.polygon(screen, RED, points)
        
        # Desenhar sensores
        for i in range(NUM_SENSORS):
            sensor_angle = self.angle - 90 + (i * 180 / (NUM_SENSORS - 1))
            rad_angle = math.radians(sensor_angle)
            
            # Calcular posição final do sensor
            end_x = self.x + self.sensors[i] * SENSOR_RANGE * math.sin(rad_angle)
            end_y = self.y - self.sensors[i] * SENSOR_RANGE * math.cos(rad_angle)
            
            pygame.draw.line(screen, GREEN, (self.x, self.y), (end_x, end_y), 1)

class QLearningAI:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.exploration_rate = EXPLORATION_RATE
        
    def get_q_value(self, state, action):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        return self.q_table[state_key][action]
    
    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            # Exploração: ação aleatória
            return random.randint(0, self.action_size - 1)
        else:
            # Exploração: melhor ação baseada na Q-table
            state_key = tuple(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_size
            return np.argmax(self.q_table[state_key])
    
    def update_q_value(self, state, action, reward, next_state):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.action_size
            
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])
        
        # Fórmula do Q-learning
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

class Track:
    def __init__(self):
        self.width = TRACK_WIDTH
        self.height = TRACK_HEIGHT
        self.center_x = WIDTH // 2
        self.center_y = HEIGHT // 2
        
    def draw(self, screen):
        # Desenhar pista
        track_rect = pygame.Rect(
            self.center_x - self.width // 2,
            self.center_y - self.height // 2,
            self.width,
            self.height
        )
        pygame.draw.rect(screen, GRAY, track_rect)
        
        # Desenhar bordas da pista
        pygame.draw.rect(screen, BLACK, track_rect, 2)

def main():
    clock = pygame.time.Clock()
    track = Track()
    car = Car(WIDTH // 2, HEIGHT // 2)
    ai = QLearningAI(NUM_SENSORS + 1, 4)  # Estado: sensores + velocidade, Ações: 4
    
    generation = 1
    best_score = -float('inf')
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Obter estado atual
        state = car.get_state()
        
        # Escolher ação
        action = ai.choose_action(state)
        
        # Aplicar ação
        car.update(action)
        
        # Calcular recompensa
        reward = 0
        
        # Recompensa por estar vivo
        if car.alive:
            reward += 0.1
            
            # Recompensa por velocidade
            reward += car.speed / MAX_SPEED * 0.1
            
            # Recompensa por manter-se no centro da pista
            track_center_x = WIDTH // 2
            track_center_y = HEIGHT // 2
            distance_from_center = math.sqrt((car.x - track_center_x)**2 + (car.y - track_center_y)**2)
            max_distance = math.sqrt((TRACK_WIDTH/2)**2 + (TRACK_HEIGHT/2)**2)
            reward += (1 - distance_from_center / max_distance) * 0.05
            
            # Penalidade por ficar muito perto das bordas
            min_sensor = min(car.sensors)
            if min_sensor < 0.2:
                reward -= 0.5
        else:
            reward -= 10
        
        # Obter próximo estado
        next_state = car.get_state()
        
        # Atualizar Q-table
        ai.update_q_value(state, action, reward, next_state)
        
        # Verificar se o carro morreu ou completou uma volta
        if not car.alive or car.total_time > 1000:
            # Calcular pontuação final
            if car.alive:
                car.score += 100  # Bônus por completar o percurso
                car.score -= car.total_time * 0.1  # Penalidade por tempo
            else:
                car.score -= 50  # Penalidade por bater
            
            # Atualizar melhor pontuação
            if car.score > best_score:
                best_score = car.score
            
            # Reiniciar carro
            car = Car(WIDTH // 2, HEIGHT // 2)
            generation += 1
        
        # Desenhar
        screen.fill(WHITE)
        track.draw(screen)
        car.draw(screen)
        
        # Desenhar informações
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Pontuação: {car.score:.2f}", True, BLACK)
        generation_text = font.render(f"Geração: {generation}", True, BLACK)
        best_score_text = font.render(f"Melhor Pontuação: {best_score:.2f}", True, BLACK)
        time_text = font.render(f"Tempo: {car.total_time}", True, BLACK)
        
        screen.blit(score_text, (10, 10))
        screen.blit(generation_text, (10, 40))
        screen.blit(best_score_text, (10, 70))
        screen.blit(time_text, (10, 100))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()