import pygame
import numpy as np
import math
import random
import sys
from typing import List, Dict, Tuple
import time

# Inicialização do Pygame
pygame.init()

# Configurações da tela
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Carrinho Autônomo - Pista Circular com Previsão")

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
GRASS_GREEN = (34, 139, 34)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
LIGHT_BLUE = (173, 216, 230)
PINK = (255, 182, 193)
COLORS = [RED, GREEN, BLUE, ORANGE, PURPLE, CYAN, YELLOW] + \
         [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(20)]

# Parâmetros da pista circular
TRACK_CENTER_X = WIDTH // 2
TRACK_CENTER_Y = HEIGHT // 2
OUTER_RADIUS = 220
INNER_RADIUS = 150
TRACK_WIDTH = OUTER_RADIUS - INNER_RADIUS

# Parâmetros do carrinho
CAR_WIDTH = 30
CAR_HEIGHT = 20
MAX_SPEED = 6
ACCELERATION = 0.3
ROTATION_SPEED = 4.0
FRICTION = 0.025

# Parâmetros da IA - OTIMIZADOS
NUM_SENSORS = 12  # Reduzido de 20
SENSOR_RANGE = 200
LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 0.6
EXPLORATION_DECAY = 0.998
MIN_EXPLORATION_RATE = 0.02

# Configuração de multi-agente
NUM_CARS = 30  # Reduzido de 50
ELITE_COUNT = 6  # Reduzido de 8
MUTATION_RATE = 0.25

# Timeouts
MAX_TIME_WITHOUT_CHECKPOINT = 300
MAX_GENERATION_TIME = 1200  # Reduzido de 1500

# Parâmetros de previsão de arcos - OTIMIZADOS
PREDICTION_TIME = 1.0  # Reduzido de 1.5
ARC_POINTS = 6  # Reduzido de 8
ARC_SEGMENTS = 4  # Reduzido de 5

# Otimização de estado
STATE_PRECISION = 1  # Casas decimais para simplificação do estado

# FUNÇÕES GLOBAIS PRIMEIRO - PARA EVITAR ERROS DE REFERÊNCIA
def circle_line_intersection(x1, y1, x2, y2, cx, cy, r):
    """Calcula interseção entre uma linha e um círculo"""
    dx = x2 - x1
    dy = y2 - y1
    
    fx = x1 - cx
    fy = y1 - cy
    
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return []
    
    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    
    intersections = []
    
    if 0 <= t1 <= 1:
        intersections.append((x1 + t1 * dx, y1 + t1 * dy))
    
    if 0 <= t2 <= 1 and abs(t1 - t2) > 1e-10:
        intersections.append((x1 + t2 * dx, y1 + t2 * dy))
    
    return intersections

def cast_ray_circular(start_x, start_y, end_x, end_y):
    """Detecção para pista circular"""
    min_distance = SENSOR_RANGE
    
    # Verificar interseção com círculo externo
    intersections_outer = circle_line_intersection(
        start_x, start_y, end_x, end_y, TRACK_CENTER_X, TRACK_CENTER_Y, OUTER_RADIUS
    )
    
    for point in intersections_outer:
        distance = math.sqrt((point[0] - start_x)**2 + (point[1] - start_y)**2)
        if distance < min_distance:
            min_distance = distance
    
    # Verificar interseção com círculo interno
    intersections_inner = circle_line_intersection(
        start_x, start_y, end_x, end_y, TRACK_CENTER_X, TRACK_CENTER_Y, INNER_RADIUS
    )
    
    for point in intersections_inner:
        distance = math.sqrt((point[0] - start_x)**2 + (point[1] - start_y)**2)
        if distance < min_distance:
            min_distance = distance
    
    return min_distance

def is_point_on_circular_track(x, y):
    """Verifica se um ponto está na pista circular"""
    distance_from_center = math.sqrt((x - TRACK_CENTER_X)**2 + (y - TRACK_CENTER_Y)**2)
    return INNER_RADIUS <= distance_from_center <= OUTER_RADIUS

def get_safe_start_position():
    """Posição inicial segura na pista circular"""
    start_angle = math.radians(-90)  # Topo da pista
    ideal_radius = (OUTER_RADIUS + INNER_RADIUS) / 2
    x = TRACK_CENTER_X + ideal_radius * math.cos(start_angle)
    y = TRACK_CENTER_Y + ideal_radius * math.sin(start_angle)
    return x, y

def get_checkpoints():
    """Cria checkpoints ao redor da pista circular"""
    checkpoints = []
    num_checkpoints = 8
    checkpoint_angle = 45  # 360/8 = 45 graus entre checkpoints
    
    ideal_radius = (OUTER_RADIUS + INNER_RADIUS) / 2
    checkpoint_width = 30
    checkpoint_height = 15
    
    for i in range(num_checkpoints):
        angle = math.radians(i * checkpoint_angle - 90)  # Começar no topo
        checkpoint_x = TRACK_CENTER_X + ideal_radius * math.cos(angle)
        checkpoint_y = TRACK_CENTER_Y + ideal_radius * math.sin(angle)
        
        # Criar retângulo orientado tangencialmente à pista
        checkpoint = pygame.Rect(
            checkpoint_x - checkpoint_width // 2,
            checkpoint_y - checkpoint_height // 2,
            checkpoint_width,
            checkpoint_height
        )
        checkpoints.append(checkpoint)
    
    return checkpoints

def is_point_in_circular_checkpoint(x, y, checkpoint):
    """Verifica se o carro está em um checkpoint"""
    return checkpoint.collidepoint(x, y)

# Métricas de evolução com performance
class Metrics:
    def __init__(self):
        self.generation_fitness = []
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.max_checkpoints_history = []
        self.survival_rate_history = []
        self.generation_times = []
        self.q_table_sizes = []
        self.processing_times = []
        
    def record_generation(self, cars, generation_time, processing_time, q_table_size):
        fitnesses = [car.fitness for car in cars]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_checkpoints = max(car.checkpoints for car in cars)
        alive_count = sum(1 for car in cars if car.alive)
        survival_rate = alive_count / len(cars)
        
        self.best_fitness_history.append(best_fitness)
        self.average_fitness_history.append(avg_fitness)
        self.max_checkpoints_history.append(max_checkpoints)
        self.survival_rate_history.append(survival_rate)
        self.generation_times.append(generation_time)
        self.q_table_sizes.append(q_table_size)
        self.processing_times.append(processing_time)
        
        return best_fitness, avg_fitness, max_checkpoints, survival_rate

metrics = Metrics()

class OptimizedQLearningAI:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.exploration_rate = EXPLORATION_RATE
        self.learning_updates = 0
        self.access_count = {}  # Contador de acessos para limpeza
        
    def simplify_state(self, state):
        """Simplifica o estado para reduzir dimensionalidade"""
        # Arredonda para menos casas decimais
        simplified = tuple(round(s, STATE_PRECISION) for s in state)
        
        # Agrupa sensores similares (cada 2 sensores viram 1)
        sensors = simplified[:NUM_SENSORS]
        grouped_sensors = []
        for i in range(0, len(sensors), 2):
            if i + 1 < len(sensors):
                grouped_sensors.append(round((sensors[i] + sensors[i+1]) / 2, STATE_PRECISION))
            else:
                grouped_sensors.append(sensors[i])
        
        # Agrupa previsões de arco
        arc_start = NUM_SENSORS
        left_arc = simplified[arc_start:arc_start + ARC_POINTS]
        right_arc = simplified[arc_start + ARC_POINTS:arc_start + 2 * ARC_POINTS]
        
        grouped_left_arc = []
        grouped_right_arc = []
        
        for i in range(0, len(left_arc), 2):
            if i + 1 < len(left_arc):
                grouped_left_arc.append(round((left_arc[i] + left_arc[i+1]) / 2, STATE_PRECISION))
            else:
                grouped_left_arc.append(left_arc[i])
                
        for i in range(0, len(right_arc), 2):
            if i + 1 < len(right_arc):
                grouped_right_arc.append(round((right_arc[i] + right_arc[i+1]) / 2, STATE_PRECISION))
            else:
                grouped_right_arc.append(right_arc[i])
        
        # Resto do estado (velocidade, aceleração, etc.)
        other_state = simplified[arc_start + 2 * ARC_POINTS:]
        
        # Estado final simplificado
        final_state = tuple(grouped_sensors + grouped_left_arc + grouped_right_arc + other_state)
        
        return final_state
    
    def get_q_value(self, state, action):
        state_simplified = self.simplify_state(state)
        
        if state_simplified not in self.q_table:
            self.q_table[state_simplified] = [0.0] * self.action_size
            self.access_count[state_simplified] = 1
        else:
            self.access_count[state_simplified] = self.access_count.get(state_simplified, 0) + 1
            
        return self.q_table[state_simplified][action]
    
    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        else:
            state_simplified = self.simplify_state(state)
            if state_simplified not in self.q_table:
                self.q_table[state_simplified] = [0.0] * self.action_size
                self.access_count[state_simplified] = 1
                return random.randint(0, self.action_size - 1)
            else:
                self.access_count[state_simplified] = self.access_count.get(state_simplified, 0) + 1
                return np.argmax(self.q_table[state_simplified])
    
    def update_q_value(self, state, action, reward, next_state):
        self.learning_updates += 1
        
        state_simplified = self.simplify_state(state)
        next_state_simplified = self.simplify_state(next_state)
        
        if state_simplified not in self.q_table:
            self.q_table[state_simplified] = [0.0] * self.action_size
            self.access_count[state_simplified] = 1
        
        if next_state_simplified not in self.q_table:
            self.q_table[next_state_simplified] = [0.0] * self.action_size
            self.access_count[next_state_simplified] = 1
            
        current_q = self.q_table[state_simplified][action]
        max_next_q = max(self.q_table[next_state_simplified])
        
        adaptive_learning_rate = self.learning_rate / (1 + self.learning_updates * 0.0001)
        
        new_q = current_q + adaptive_learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_simplified][action] = new_q
        
        # Atualizar contador de acesso
        self.access_count[state_simplified] = self.access_count.get(state_simplified, 0) + 1
    
    def cleanup_q_table(self, min_access_count=2):
        """Remove estados raramente usados para otimizar memória"""
        if len(self.q_table) > 10000:  # Só limpa se a tabela estiver muito grande
            states_to_remove = []
            for state, count in self.access_count.items():
                if count <= min_access_count:
                    states_to_remove.append(state)
            
            for state in states_to_remove:
                if state in self.q_table:
                    del self.q_table[state]
                if state in self.access_count:
                    del self.access_count[state]
    
    def crossover(self, other):
        new_brain = OptimizedQLearningAI(self.state_size, self.action_size)
        new_brain.q_table = {}
        new_brain.access_count = {}
        
        # Combina apenas os estados mais importantes
        all_states = set()
        
        # Pega os estados mais acessados de cada pai
        parent1_states = sorted(self.access_count.items(), key=lambda x: x[1], reverse=True)[:500]
        parent2_states = sorted(other.access_count.items(), key=lambda x: x[1], reverse=True)[:500]
        
        for state, _ in parent1_states + parent2_states:
            all_states.add(state)
        
        for state in all_states:
            if state in self.q_table and state in other.q_table:
                # Combinação ponderada
                new_brain.q_table[state] = [
                    (self.q_table[state][i] * 0.6 + other.q_table[state][i] * 0.4)
                    for i in range(self.action_size)
                ]
            elif state in self.q_table:
                new_brain.q_table[state] = self.q_table[state][:]
            elif state in other.q_table:
                new_brain.q_table[state] = other.q_table[state][:]
            
            new_brain.access_count[state] = 1
        
        if random.random() < MUTATION_RATE:
            self.mutate(new_brain)
        
        new_brain.exploration_rate = max(MIN_EXPLORATION_RATE, 
                                        (self.exploration_rate + other.exploration_rate) / 2 * EXPLORATION_DECAY)
        new_brain.learning_updates = (self.learning_updates + other.learning_updates) // 2
        
        return new_brain
    
    def mutate(self, brain):
        if brain.q_table:
            # Muta apenas alguns estados aleatórios
            num_mutations = min(8, len(brain.q_table) // 20)  # Menos mutações
            if num_mutations > 0:
                random_states = random.sample(list(brain.q_table.keys()), num_mutations)
                for state in random_states:
                    for i in range(self.action_size):
                        if random.random() < 0.3:  # Probabilidade reduzida
                            mutation = random.uniform(-0.3, 0.3)  # Mutação menor
                            brain.q_table[state][i] += mutation

class Car:
    def __init__(self, x=None, y=None, color=None, brain=None):
        if x is None or y is None:
            x, y = get_safe_start_position()
        
        self.x = x
        self.y = y
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.angle = -90  # Virado para CIMA
        self.speed = 0
        self.acceleration = 0
        self.sensors = [0] * NUM_SENSORS
        self.alive = True
        self.score = 0
        self.checkpoints = 0
        self.total_time = 0
        self.last_checkpoint_time = 0
        self.distance_traveled = 0
        self.color = color if color else random.choice(COLORS)
        
        # Novos sensores de previsão
        self.left_arc_prediction = [0] * ARC_POINTS
        self.right_arc_prediction = [0] * ARC_POINTS
        self.arc_collision_risk = 0.0
        
        # Estado reduzido após simplificação
        simplified_sensors = NUM_SENSORS // 2
        simplified_arcs = (ARC_POINTS // 2) * 2
        state_size = simplified_sensors + simplified_arcs + 5  # Estado simplificado
        
        self.brain = brain if brain else OptimizedQLearningAI(state_size, 5)
        
        self.fitness = 0
        self.last_checkpoint_id = -1
        self.stuck_timer = 0
        self.last_x = x
        self.last_y = y
        self.collision_count = 0
        self.best_checkpoint = 0
        self.center_bonus = 0
        self.last_speed = 0
        
    def update(self):
        if not self.alive:
            return
        
        # Detecção de travamento - menos frequente
        if self.total_time % 45 == 0:  # Aumentado de 30 para 45
            distance_moved = math.sqrt((self.x - self.last_x)**2 + (self.y - self.last_y)**2)
            if distance_moved < 8:
                self.stuck_timer += 1
                if self.stuck_timer > 4:  # Reduzido de 6 para 4
                    self.alive = False
                    self.score -= 80
                    return
            else:
                self.stuck_timer = 0
            self.last_x = self.x
            self.last_y = self.y
        
        # Timeout por checkpoint
        if self.total_time - self.last_checkpoint_time > MAX_TIME_WITHOUT_CHECKPOINT:
            self.alive = False
            self.score -= 40
            return
            
        # Atualizar previsões de arco primeiro
        self.update_arc_predictions()
        
        state = self.get_state()
        action = self.brain.choose_action(state)
        
        # Aplicar ação
        if action == 0:  # Acelerar forte
            self.speed = min(self.speed + ACCELERATION * 1.5, MAX_SPEED)
            self.acceleration = ACCELERATION * 1.5
        elif action == 1:  # Acelerar suave
            self.speed = min(self.speed + ACCELERATION, MAX_SPEED)
            self.acceleration = ACCELERATION
        elif action == 2:  # Frear
            self.speed = max(self.speed - ACCELERATION * 2.0, -MAX_SPEED/3)
            self.acceleration = -ACCELERATION * 2.0
        elif action == 3:  # Virar à esquerda
            if abs(self.speed) > 0.3:
                self.angle -= ROTATION_SPEED * (0.3 + abs(self.speed)/MAX_SPEED)
                self.acceleration = 0
        elif action == 4:  # Virar à direita
            if abs(self.speed) > 0.3:
                self.angle += ROTATION_SPEED * (0.3 + abs(self.speed)/MAX_SPEED)
                self.acceleration = 0
            
        # Aplicar atrito
        if self.speed > 0:
            self.speed = max(self.speed - FRICTION * (1 + abs(self.speed)/MAX_SPEED), 0)
        elif self.speed < 0:
            self.speed = min(self.speed + FRICTION * (1 + abs(self.speed)/MAX_SPEED), 0)
            
        # Movimentar o carro
        rad_angle = math.radians(self.angle)
        new_x = self.x + self.speed * math.cos(rad_angle)
        new_y = self.y + self.speed * math.sin(rad_angle)
        
        # Atualizar distância percorrida
        distance_step = math.sqrt((new_x - self.x)**2 + (new_y - self.y)**2)
        self.distance_traveled += distance_step
        
        self.x = new_x
        self.y = new_y
        
        # Calcular aceleração real
        if self.total_time > 0:
            self.acceleration = self.speed - self.last_speed
        self.last_speed = self.speed
        
        # Atualizar sensores - menos frequente
        if self.total_time % 2 == 0:  # Atualiza a cada 2 frames
            self.update_sensors()
        
        # Verificação de colisão
        collision_before = self.alive
        self.check_collision_with_perimeter()
        if collision_before and not self.alive:
            self.collision_count += 1
        
        # Verificar checkpoints
        checkpoint_before = self.checkpoints
        self.check_checkpoints()
        checkpoint_gained = self.checkpoints > checkpoint_before
        
        # Recompensa por estar no centro da pista
        center_distance = self.distance_to_track_center()
        if center_distance < 25:
            self.center_bonus += 0.2
            self.score += 0.2
        
        # Atualizar tempo
        self.total_time += 1
        
        # Aprendizado com recompensas detalhadas - menos frequente
        if self.total_time % 3 == 0:  # Aprende a cada 3 frames
            next_state = self.get_state()
            reward = self.calculate_detailed_reward(checkpoint_gained, self.speed, distance_step, center_distance)
            self.brain.update_q_value(state, action, reward, next_state)
    
    def update_arc_predictions(self):
        """Atualiza as previsões de arco para os vértices do carro"""
        if abs(self.speed) < 0.1:
            self.left_arc_prediction = [1.0] * ARC_POINTS
            self.right_arc_prediction = [1.0] * ARC_POINTS
            self.arc_collision_risk = 0.0
            return
        
        front_left, front_right = self.get_front_vertices()
        
        time_step = PREDICTION_TIME / ARC_POINTS
        left_collision_count = 0
        right_collision_count = 0
        
        for i in range(ARC_POINTS):
            future_time = (i + 1) * time_step
            distance = self.speed * future_time
            
            left_future_x = front_left[0] + distance * math.cos(math.radians(self.angle))
            left_future_y = front_left[1] + distance * math.sin(math.radians(self.angle))
            
            right_future_x = front_right[0] + distance * math.cos(math.radians(self.angle))
            right_future_y = front_right[1] + distance * math.sin(math.radians(self.angle))
            
            left_safe = is_point_on_circular_track(left_future_x, left_future_y)
            right_safe = is_point_on_circular_track(right_future_x, right_future_y)
            
            self.left_arc_prediction[i] = 1.0 if left_safe else 0.0
            self.right_arc_prediction[i] = 1.0 if right_safe else 0.0
            
            if not left_safe:
                left_collision_count += 1
            if not right_safe:
                right_collision_count += 1
        
        total_points = ARC_POINTS * 2
        collision_points = left_collision_count + right_collision_count
        self.arc_collision_risk = collision_points / total_points

    def get_front_vertices(self):
        """Retorna os vértices frontais do carro"""
        rad_angle = math.radians(self.angle)
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)
        
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Vértice frontal esquerdo
        front_left = (
            self.x + half_width * cos_a - half_height * sin_a,
            self.y + half_width * sin_a + half_height * cos_a
        )
        
        # Vértice frontal direito
        front_right = (
            self.x - half_width * cos_a - half_height * sin_a,
            self.y - half_width * sin_a + half_height * cos_a
        )
        
        return front_left, front_right
    
    def distance_to_track_center(self):
        """Calcula a distância do carro até o centro ideal da pista"""
        distance_from_center = math.sqrt((self.x - TRACK_CENTER_X)**2 + (self.y - TRACK_CENTER_Y)**2)
        ideal_distance = (OUTER_RADIUS + INNER_RADIUS) / 2
        
        # Calcular distâncias ideais baseadas na largura
        track_center_distance = abs(distance_from_center - ideal_distance)
        
        # Distâncias ideais conforme especificado
        ideal_margin = TRACK_WIDTH / 4  # x/4
        close_margin = TRACK_WIDTH / 8  # x/8
        car_half_width = CAR_WIDTH / 2
        
        # Ajustar para considerar a largura do carro
        effective_distance = track_center_distance - car_half_width
        
        if effective_distance < close_margin:
            # Muito próximo da borda
            return effective_distance + 20  # Penalidade extra
        elif effective_distance < ideal_margin:
            # Dentro da margem aceitável
            return effective_distance
        else:
            # Muito centralizado (também não é ideal)
            return effective_distance
    
    def calculate_detailed_reward(self, checkpoint_gained, old_speed, distance_step, center_distance):
        reward = 0
        
        if checkpoint_gained:
            reward += 300
            speed_bonus = abs(self.speed) * 25
            reward += speed_bonus
        
        if self.alive:
            reward += distance_step * 0.8
            
            ideal_speed_range = (MAX_SPEED * 0.4, MAX_SPEED * 0.8)
            if ideal_speed_range[0] < abs(self.speed) < ideal_speed_range[1]:
                reward += 2
            elif abs(self.speed) < MAX_SPEED * 0.2:
                reward -= 1
            
            # Recompensa baseada na distância ideal
            ideal_margin = TRACK_WIDTH / 4
            close_margin = TRACK_WIDTH / 8
            
            if center_distance < close_margin:
                reward -= 5  # Muito perto da borda
            elif center_distance < ideal_margin:
                reward += 2  # Posição ideal
            else:
                reward += 1  # Muito centralizado
            
            if abs(self.acceleration) < ACCELERATION * 0.5:
                reward += 0.5
        
        # Penalidades baseadas nas previsões de arco
        if self.arc_collision_risk > 0.7:
            reward -= 15  # Alto risco de colisão futura
        elif self.arc_collision_risk > 0.4:
            reward -= 5   # Risco moderado
        
        min_sensor = min(self.sensors) if self.sensors else 1
        if min_sensor < 0.1:
            reward -= 8
        elif min_sensor < 0.2:
            reward -= 3
        
        if not self.alive and self.collision_count > 0:
            reward -= 150
        
        if abs(self.acceleration) > ACCELERATION * 3:
            reward -= 2
        
        return reward
        
    def update_sensors(self):
        """Atualiza os sensores para detecção na pista circular"""
        for i in range(NUM_SENSORS):
            sensor_angle = self.angle + (i * 360 / NUM_SENSORS)
            rad_angle = math.radians(sensor_angle)
            
            end_x = self.x + SENSOR_RANGE * math.cos(rad_angle)
            end_y = self.y + SENSOR_RANGE * math.sin(rad_angle)
            
            distance = cast_ray_circular(self.x, self.y, end_x, end_y)
            self.sensors[i] = distance / SENSOR_RANGE
    
    def check_collision_with_perimeter(self):
        """Verifica colisão com a pista circular"""
        car_points = self.get_car_perimeter_points()
        
        for point in car_points:
            if not is_point_on_circular_track(point[0], point[1]):
                self.alive = False
                self.score -= 120
                return
    
    def get_car_perimeter_points(self):
        """Retorna pontos distribuídos pelo perímetro do carro"""
        rad_angle = math.radians(self.angle)
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)
        
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Pontos dos vértices do carro
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
        
        # Adicionar pontos intermediários nas bordas para melhor detecção
        perimeter_points = []
        
        # Pontos ao longo de cada borda
        for i in range(4):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % 4]
            
            # Adicionar pontos intermediários
            for j in range(3):
                t = (j + 1) / 4
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                perimeter_points.append((px, py))
        
        # Incluir os vértices também
        perimeter_points.extend(points)
        
        return perimeter_points
    
    def check_checkpoints(self):
        checkpoints = get_checkpoints()
        current_time = self.total_time
        
        for i, checkpoint in enumerate(checkpoints):
            next_checkpoint = (self.last_checkpoint_id + 1) % len(checkpoints)
            
            if (is_point_in_circular_checkpoint(self.x, self.y, checkpoint) and i == next_checkpoint):
                self.checkpoints += 1
                self.score += 250
                self.last_checkpoint_time = current_time
                self.last_checkpoint_id = i
                
                if self.checkpoints > self.best_checkpoint:
                    self.best_checkpoint = self.checkpoints
                
                speed_bonus = abs(self.speed) * 15
                self.score += speed_bonus
                break
    
    def calculate_fitness(self):
        fitness = 0
        
        fitness += self.checkpoints * 1200
        fitness += self.best_checkpoint * 600
        fitness += self.distance_traveled * 0.2
        fitness += self.center_bonus * 8
        fitness += self.score * 1.2
        
        fitness -= self.collision_count * 60
        
        # Bonus por usar previsões eficientemente
        if self.arc_collision_risk < 0.2:
            fitness += 50
        
        if self.total_time > 200:
            fitness += self.total_time * 0.1
        
        if self.total_time > 0:
            avg_speed = self.distance_traveled / self.total_time
            if avg_speed > 2.0:
                fitness += avg_speed * 20
        
        if self.total_time < 100:
            fitness *= 0.4
        elif self.total_time < 200:
            fitness *= 0.7
        
        if self.checkpoints >= 4:
            laps = self.checkpoints // 4
            fitness *= (1.8 ** laps)
        
        return max(1, fitness)
    
    def get_state(self):
        """Retorna o estado simplificado para a IA"""
        speed_normalized = self.speed / MAX_SPEED
        acceleration_normalized = self.acceleration / (ACCELERATION * 2)
        angle_rad = math.radians(self.angle)
        
        # O método simplify_state da IA vai processar isso
        return (self.sensors + 
                self.left_arc_prediction + 
                self.right_arc_prediction + 
                [
                    speed_normalized,
                    acceleration_normalized, 
                    math.sin(angle_rad), 
                    math.cos(angle_rad),
                    self.arc_collision_risk
                ])
    
    def draw(self, screen):
        if not self.alive:
            return
            
        rad_angle = math.radians(self.angle)
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)
        
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
        
        pygame.draw.polygon(screen, self.color, points)
        
        if self == cars[0]:
            # Desenhar direção
            direction_x = self.x + half_height * 2.0 * cos_a
            direction_y = self.y + half_height * 2.0 * sin_a
            pygame.draw.line(screen, WHITE, (self.x, self.y), (direction_x, direction_y), 3)
            
            # Desenhar sensores
            for i in range(0, NUM_SENSORS, 2):
                sensor_angle = self.angle + (i * 360 / NUM_SENSORS)
                rad_angle = math.radians(sensor_angle)
                
                sensor_distance = self.sensors[i] * SENSOR_RANGE
                end_x = self.x + sensor_distance * math.cos(rad_angle)
                end_y = self.y + sensor_distance * math.sin(rad_angle)
                
                if self.sensors[i] < 0.2:
                    color = RED
                elif self.sensors[i] < 0.4:
                    color = YELLOW
                else:
                    color = GREEN
                
                pygame.draw.line(screen, color, (self.x, self.y), (end_x, end_y), 1)
            
            # Desenhar arcos de previsão
            self.draw_prediction_arcs(screen)
    
    def draw_prediction_arcs(self, screen):
        """Desenha os arcos de previsão para visualização"""
        if abs(self.speed) < 0.1:
            return
            
        front_left, front_right = self.get_front_vertices()
        time_step = PREDICTION_TIME / ARC_SEGMENTS
        
        # Desenhar arco esquerdo
        left_points = []
        for i in range(ARC_SEGMENTS):
            t = (i + 1) / ARC_SEGMENTS
            future_time = PREDICTION_TIME * t
            distance = self.speed * future_time
            
            future_x = front_left[0] + distance * math.cos(math.radians(self.angle))
            future_y = front_left[1] + distance * math.sin(math.radians(self.angle))
            left_points.append((future_x, future_y))
        
        if len(left_points) > 1:
            arc_color = LIGHT_BLUE if self.left_arc_prediction[-1] > 0.5 else PINK
            pygame.draw.lines(screen, arc_color, False, left_points, 2)
        
        # Desenhar arco direito
        right_points = []
        for i in range(ARC_SEGMENTS):
            t = (i + 1) / ARC_SEGMENTS
            future_time = PREDICTION_TIME * t
            distance = self.speed * future_time
            
            future_x = front_right[0] + distance * math.cos(math.radians(self.angle))
            future_y = front_right[1] + distance * math.sin(math.radians(self.angle))
            right_points.append((future_x, future_y))
        
        if len(right_points) > 1:
            arc_color = LIGHT_BLUE if self.right_arc_prediction[-1] > 0.5 else PINK
            pygame.draw.lines(screen, arc_color, False, right_points, 2)

def draw_track(screen):
    """Desenha a pista circular"""
    screen.fill(GRASS_GREEN)
    
    # Desenhar pista (área entre os círculos)
    pygame.draw.circle(screen, DARK_GRAY, (TRACK_CENTER_X, TRACK_CENTER_Y), OUTER_RADIUS)
    pygame.draw.circle(screen, GRASS_GREEN, (TRACK_CENTER_X, TRACK_CENTER_Y), INNER_RADIUS)
    
    # Bordas da pista
    pygame.draw.circle(screen, WHITE, (TRACK_CENTER_X, TRACK_CENTER_Y), OUTER_RADIUS, 3)
    pygame.draw.circle(screen, WHITE, (TRACK_CENTER_X, TRACK_CENTER_Y), INNER_RADIUS, 3)
    
    # Linha de partida
    start_angle = math.radians(-100)
    end_angle = math.radians(-80)
    pygame.draw.arc(screen, WHITE, 
                   (TRACK_CENTER_X - OUTER_RADIUS, TRACK_CENTER_Y - OUTER_RADIUS, 
                    OUTER_RADIUS * 2, OUTER_RADIUS * 2),
                   start_angle, end_angle, 5)
    
    # Checkpoints
    checkpoints = get_checkpoints()
    for i, checkpoint in enumerate(checkpoints):
        color = YELLOW if i == 0 else ORANGE
        pygame.draw.rect(screen, color, checkpoint)
        
        # Número do checkpoint
        font = pygame.font.SysFont(None, 20)
        text = font.render(str(i + 1), True, BLACK)
        text_rect = text.get_rect(center=checkpoint.center)
        screen.blit(text, text_rect)

def create_new_generation():
    global cars, generation, best_cars
    
    start_time = time.time()
    
    # Calcular fitness para todos os carros
    for car in cars:
        car.fitness = car.calculate_fitness()
    
    # Ordenar por fitness
    cars.sort(key=lambda x: x.fitness, reverse=True)
    
    # Limpar Q-tables dos melhores carros para otimizar
    for car in cars[:ELITE_COUNT]:
        car.brain.cleanup_q_table()
    
    # Calcular tamanho total da Q-table
    total_q_size = sum(len(car.brain.q_table) for car in cars[:ELITE_COUNT])
    
    # Registrar métricas da geração
    generation_time = cars[0].total_time if cars else 0
    processing_time = time.time() - start_time
    best_fitness, avg_fitness, max_checkpoints, survival_rate = metrics.record_generation(
        cars, generation_time, processing_time, total_q_size
    )
    
    print(f"Geração {generation}:")
    print(f"  Melhor Fitness: {best_fitness:.1f}")
    print(f"  Fitness Médio: {avg_fitness:.1f}")
    print(f"  Máximo Checkpoints: {max_checkpoints}")
    print(f"  Taxa de Sobrevivência: {survival_rate:.1%}")
    print(f"  Tempo da Geração: {generation_time}")
    print(f"  Tamanho Q-table: {total_q_size} estados")
    print(f"  Tempo Processamento: {processing_time:.2f}s")
    print("-" * 50)
    
    # Manter os melhores
    best_cars = cars[:ELITE_COUNT]
    
    # Criar nova geração
    new_cars = []
    
    # Manter a elite
    for i in range(ELITE_COUNT):
        elite_car = best_cars[i]
        new_car = Car(
            x=get_safe_start_position()[0],
            y=get_safe_start_position()[1],
            color=elite_car.color,
            brain=elite_car.brain
        )
        new_car.brain.exploration_rate = max(MIN_EXPLORATION_RATE, 
                                           elite_car.brain.exploration_rate * EXPLORATION_DECAY)
        new_cars.append(new_car)
    
    # Cruzamento e mutação mais eficiente
    while len(new_cars) < NUM_CARS:
        parent1 = random.choice(best_cars[:4])  # Usa apenas os 4 melhores
        parent2 = random.choice(best_cars[:4])
        
        child_brain = parent1.brain.crossover(parent2.brain)
        new_car = Car(
            x=get_safe_start_position()[0],
            y=get_safe_start_position()[1],
            brain=child_brain
        )
        new_cars.append(new_car)
    
    cars = new_cars
    generation += 1

def draw_ui(screen, generation, best_score, alive_count, time_elapsed):
    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 22)
    
    # Informações da geração
    gen_text = font.render(f"Geração: {generation}", True, WHITE)
    screen.blit(gen_text, (10, 10))
    
    # Melhor score
    best_text = font.render(f"Melhor Score: {best_score:.1f}", True, WHITE)
    screen.blit(best_text, (10, 40))
    
    # Carros vivos
    alive_text = font.render(f"Carros Vivos: {alive_count}/{NUM_CARS}", True, WHITE)
    screen.blit(alive_text, (10, 70))
    
    # Tempo
    time_text = font.render(f"Tempo: {time_elapsed}", True, WHITE)
    screen.blit(time_text, (10, 100))
    
    # Performance
    if generation > 1 and len(metrics.q_table_sizes) > 0:
        perf_text = small_font.render(f"Q-table: {metrics.q_table_sizes[-1]} estados", True, WHITE)
        screen.blit(perf_text, (10, 130))
    
    # Métricas de evolução (se disponíveis)
    if generation > 1 and len(metrics.best_fitness_history) >= 2:
        improvement = metrics.best_fitness_history[-1] - metrics.best_fitness_history[-2]
        improvement_text = small_font.render(f"Melhora: {improvement:+.1f}", True, GREEN if improvement > 0 else RED)
        screen.blit(improvement_text, (10, 155))
        
        avg_fitness = metrics.average_fitness_history[-1]
        avg_text = small_font.render(f"Fitness Médio: {avg_fitness:.1f}", True, WHITE)
        screen.blit(avg_text, (10, 180))
    
    # Controles
    controls_text = small_font.render("P: Pausar  R: Reiniciar  ESPAÇO: Nova Geração", True, WHITE)
    screen.blit(controls_text, (WIDTH - 350, 10))
    
    # Estatísticas do carro líder (se houver)
    if cars:
        leader = cars[0]
        if leader.alive:
            speed_text = small_font.render(f"Velocidade: {abs(leader.speed):.1f}", True, WHITE)
            screen.blit(speed_text, (WIDTH - 200, 40))
            
            checkpoints_text = small_font.render(f"Checkpoints: {leader.checkpoints}", True, WHITE)
            screen.blit(checkpoints_text, (WIDTH - 200, 65))
            
            fitness_text = small_font.render(f"Fitness: {leader.fitness:.1f}", True, WHITE)
            screen.blit(fitness_text, (WIDTH - 200, 90))
            
            # Mostrar risco de colisão previsto
            risk_text = small_font.render(f"Risco: {leader.arc_collision_risk:.1%}", True, 
                                         RED if leader.arc_collision_risk > 0.5 else GREEN)
            screen.blit(risk_text, (WIDTH - 200, 115))

# Variáveis globais
cars = []
generation = 1
best_cars = []
paused = False
clock = pygame.time.Clock()

# Inicializar primeira geração
for _ in range(NUM_CARS):
    cars.append(Car())

# Loop principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                paused = not paused
            elif event.key == pygame.K_r:
                # Reiniciar simulação
                cars = []
                generation = 1
                metrics = Metrics()
                for _ in range(NUM_CARS):
                    cars.append(Car())
            elif event.key == pygame.K_SPACE:
                create_new_generation()
    
    if not paused:
        alive_count = 0
        for car in cars:
            if car.alive:
                car.update()
                alive_count += 1
        
        all_dead = alive_count == 0
        time_expired = cars[0].total_time > MAX_GENERATION_TIME if cars else False
        
        if all_dead or time_expired:
            create_new_generation()
    
    # Desenho
    draw_track(screen)
    for car in cars:
        car.draw(screen)
    
    best_score = 0
    for car in cars:
        if car.alive and car.score > best_score:
            best_score = car.score
    
    time_elapsed = cars[0].total_time if cars else 0
    draw_ui(screen, generation, best_score, alive_count, time_elapsed)
    
    if paused:
        font = pygame.font.SysFont(None, 48)
        pause_text = font.render("PAUSADO", True, RED)
        text_rect = pause_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(pause_text, text_rect)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()