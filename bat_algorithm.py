import numpy as np
import random

class BatAlgorithm:
    def __init__(self, 
                 population_size=100, 
                 dimensions=None,
                 bounds=(-5.0, 5.0),
                 max_iterations=1000,
                 alpha=0.9,
                 gamma=0.9,
                 fmin=0,
                 fmax=100):
        """
        Algoritmo dos Morcegos (Bat Algorithm)
        
        Args:
            population_size: Tamanho da população de morcegos
            dimensions: Número de dimensões do problema (tamanho do vetor de pesos)
            bounds: Limites inferior e superior para as variáveis
            max_iterations: Número máximo de iterações
            alpha: Fator de redução da sonoridade
            gamma: Fator de aumento da taxa de pulsos
            fmin: Frequência mínima
            fmax: Frequência máxima
        """
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.gamma = gamma
        self.fmin = fmin
        self.fmax = fmax
        
        # Inicialização das populações
        self.population = None
        self.velocity = None
        self.frequency = None
        self.loudness = None
        self.pulse_rate = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = -np.inf
        
        self.iteration = 0
        self.fitness_history = []
    
    def initialize_population(self):
        """Inicializa a população de morcegos"""
        # Posições dos morcegos (soluções)
        self.population = np.random.uniform(
            self.bounds[0], self.bounds[1], 
            (self.population_size, self.dimensions)
        )
        
        # Velocidades iniciais
        self.velocity = np.zeros((self.population_size, self.dimensions))
        
        # Frequências aleatórias
        self.frequency = np.random.uniform(
            self.fmin, self.fmax, self.population_size
        )
        
        # Sonoridade inicial (A0)
        self.loudness = np.random.uniform(1.0, 2.0, self.population_size)
        
        # Taxa de emissão de pulsos inicial (r0)
        self.pulse_rate = np.random.uniform(0.0, 1.0, self.population_size)
        
        # Fitness inicial
        self.fitness = np.full(self.population_size, -np.inf)
    
    def update_frequency_and_velocity(self, i, best_position):
        """Atualiza frequência e velocidade do morcego i"""
        # Atualizar frequência
        beta = np.random.random()
        self.frequency[i] = self.fmin + (self.fmax - self.fmin) * beta
        
        # Atualizar velocidade
        self.velocity[i] = (self.velocity[i] + 
                           (self.population[i] - best_position) * self.frequency[i])
    
    def update_position(self, i):
        """Atualiza posição do morcego i"""
        self.population[i] = self.population[i] + self.velocity[i]
        
        # Aplicar limites
        self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])
    
    def local_search(self, best_position):
        """Busca local ao redor da melhor solução"""
        epsilon = np.random.uniform(-1, 1, self.dimensions)
        avg_loudness = np.mean(self.loudness)
        new_solution = best_position + epsilon * avg_loudness
        
        # Aplicar limites
        new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
        return new_solution
    
    def update_loudness_and_pulse_rate(self, i):
        """Atualiza sonoridade e taxa de pulsos do morcego i"""
        self.loudness[i] = self.alpha * self.loudness[i]
        self.pulse_rate[i] = self.pulse_rate[i] * (1 - np.exp(-self.gamma * self.iteration))
    
    def evolve(self, fitness_function, parallel=False):
        """
        Uma iteração do algoritmo
        
        Args:
            fitness_function: Função de fitness
            parallel: Se True, avalia toda população de uma vez
            
        Returns:
            best_solution, best_fitness da iteração atual
        """
        if self.population is None:
            self.initialize_population()
        
        # Avaliar fitness da população
        if parallel:
            self.fitness = fitness_function(self.population)
        else:
            for i in range(self.population_size):
                self.fitness[i] = fitness_function(self.population[i])
        
        # Encontrar melhor solução atual
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()
        
        # Atualizar cada morcego
        for i in range(self.population_size):
            # Gerar nova solução ajustando frequência e velocidade
            self.update_frequency_and_velocity(i, self.best_solution)
            self.update_position(i)
            
            # Busca local com probabilidade baseada na taxa de pulso
            if np.random.random() > self.pulse_rate[i]:
                new_solution = self.local_search(self.best_solution)
                
                # Avaliar nova solução
                if parallel:
                    # Para implementação paralela, usar a população atual
                    pass
                else:
                    new_fitness = fitness_function(new_solution)
                    
                    # Aceitar nova solução se for melhor e com probabilidade baseada na sonoridade
                    if (new_fitness > self.fitness[i] and 
                        np.random.random() < self.loudness[i]):
                        self.population[i] = new_solution.copy()
                        self.fitness[i] = new_fitness
                        
                        # Atualizar sonoridade e taxa de pulsos
                        self.update_loudness_and_pulse_rate(i)
        
        self.iteration += 1
        self.fitness_history.append(self.best_fitness)
        
        return self.best_solution.copy(), self.best_fitness
    
    def optimize(self, fitness_function, parallel=False):
        """
        Executa o algoritmo completo
        
        Args:
            fitness_function: Função de fitness a ser otimizada
            parallel: Se True, usa avaliação paralela
            
        Returns:
            best_solution, best_fitness, fitness_history
        """
        self.initialize_population()
        
        for iteration in range(self.max_iterations):
            self.evolve(fitness_function, parallel)
            
            # Log do progresso
            if iteration % 10 == 0:
                print(f"Iteração {iteration}: Melhor Fitness = {self.best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness, self.fitness_history
    
    def get_population(self):
        """Retorna a população atual"""
        return self.population.copy() if self.population is not None else None
    
    def set_dimensions(self, dimensions):
        """Define o número de dimensões do problema"""
        self.dimensions = dimensions