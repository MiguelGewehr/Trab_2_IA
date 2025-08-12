import numpy as np
import sys
import os
import time
from typing import List

# Adicionar o diretório game ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.core import SurvivalGame, GameConfig
from neural_network import NeuralNetworkAgent
from bat_algorithm import BatAlgorithm

class GameTrainer:
    def __init__(self, 
                 population_size=100,
                 max_iterations=1000,
                 time_limit_hours=12):
        """
        Treinador do agente usando Bat Algorithm + Neural Network
        
        Args:
            population_size: Tamanho da população de morcegos
            max_iterations: Número máximo de iterações
            time_limit_hours: Limite de tempo em horas
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.time_limit = time_limit_hours * 3600  # Converter para segundos
        
        # Configurações do jogo
        self.game_config = GameConfig(
            num_players=population_size,
            fps=60,
            render_grid=False
        )
        
        # Criar agente modelo para obter dimensões
        self.model_agent = NeuralNetworkAgent(
            input_size=27,  # 25 (grade 5x5) + 2 (variáveis internas)
            hidden_layers=[32, 16],
            output_size=3
        )
        
        # Obter dimensões dos pesos
        self.weight_dimensions = self.model_agent.get_weights_size()
        print(f"Dimensões do vetor de pesos: {self.weight_dimensions}")
        
        # Inicializar Bat Algorithm
        self.bat_algorithm = BatAlgorithm(
            population_size=population_size,
            dimensions=self.weight_dimensions,
            bounds=(-2.0, 2.0),  # Limites para os pesos
            max_iterations=max_iterations,
            alpha=0.9,
            gamma=0.9,
            fmin=0,
            fmax=100
        )
        
        # Variáveis de controle
        self.start_time = None
        self.best_fitness_history = []
        self.best_weights = None
        self.best_fitness = -np.inf
    
    def fitness_function_single(self, weights: np.ndarray) -> float:
        """
        Função de fitness para um único agente
        Avalia o desempenho de um agente no jogo
        """
        agent = NeuralNetworkAgent()
        agent.set_weights(weights)
        
        total_score = 0
        num_games = 3  # Jogar 3 vezes para obter média
        
        for game_num in range(num_games):
            game_config = GameConfig(num_players=1, render_grid=False)
            game = SurvivalGame(config=game_config, render=False)
            
            while not game.all_players_dead():
                # Verificar limite de tempo
                if self.start_time and (time.time() - self.start_time) > self.time_limit:
                    return total_score / max(1, game_num)
                
                state = game.get_state(0, include_internals=True)
                action = agent.predict(state)
                game.update([action])
            
            total_score += game.players[0].score
        
        return total_score / num_games
    
    def fitness_function_parallel(self, population: np.ndarray) -> np.ndarray:
        """
        Função de fitness paralela - avalia toda a população de uma vez
        """
        # Criar agentes para toda a população
        agents = []
        for weights in population:
            agent = NeuralNetworkAgent()
            agent.set_weights(weights)
            agents.append(agent)
        
        total_scores = np.zeros(len(agents))
        num_games = 3
        
        for game_num in range(num_games):
            # Verificar limite de tempo
            if self.start_time and (time.time() - self.start_time) > self.time_limit:
                return total_scores / max(1, game_num)
            
            game = SurvivalGame(config=self.game_config, render=False)
            
            while not game.all_players_dead():
                actions = []
                for idx, agent in enumerate(agents):
                    if game.players[idx].alive:
                        state = game.get_state(idx, include_internals=True)
                        action = agent.predict(state)
                        actions.append(action)
                    else:
                        actions.append(0)
                
                game.update(actions)
            
            # Somar scores
            for idx, player in enumerate(game.players):
                total_scores[idx] += player.score
        
        # Retornar médias
        average_scores = total_scores / num_games
        
        # Log do progresso
        print(f"Melhor: {np.max(average_scores):.2f} | "
              f"Média: {np.mean(average_scores):.2f} | "
              f"Std: {np.std(average_scores):.2f}")
        
        return average_scores
    
    def train(self, use_parallel=True):
        """
        Executa o treinamento principal
        """
        print("=== INICIANDO TREINAMENTO ===")
        print(f"População: {self.population_size}")
        print(f"Iterações máximas: {self.max_iterations}")
        print(f"Dimensões dos pesos: {self.weight_dimensions}")
        print(f"Limite de tempo: {self.time_limit/3600:.1f} horas")
        print(f"Modo paralelo: {use_parallel}")
        
        self.start_time = time.time()
        
        try:
            for iteration in range(self.max_iterations):
                # Verificar limite de tempo
                elapsed_time = time.time() - self.start_time
                if elapsed_time > self.time_limit:
                    print(f"\nLimite de tempo atingido: {elapsed_time/3600:.2f} horas")
                    break
                
                iteration_start = time.time()
                
                # Executar uma iteração do Bat Algorithm
                if use_parallel:
                    current_best_weights, current_best_fitness = self.bat_algorithm.evolve(
                        self.fitness_function_parallel, parallel=True
                    )
                else:
                    current_best_weights, current_best_fitness = self.bat_algorithm.evolve(
                        self.fitness_function_single, parallel=False
                    )
                
                # Atualizar melhor solução global
                if current_best_fitness > self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_weights = current_best_weights.copy()
                    
                    # Salvar backup
                    np.save("best_weights_backup.npy", self.best_weights)
                    print(f"Novo melhor fitness: {self.best_fitness:.2f}")
                
                self.best_fitness_history.append(self.best_fitness)
                
                iteration_time = time.time() - iteration_start
                elapsed_total = time.time() - self.start_time
                
                print(f"Iteração {iteration+1}/{self.max_iterations} | "
                      f"Fitness Atual: {current_best_fitness:.2f} | "
                      f"Melhor Global: {self.best_fitness:.2f} | "
                      f"Tempo: {iteration_time:.1f}s | "
                      f"Total: {elapsed_total/60:.1f}min")
        
        except KeyboardInterrupt:
            print("\nTreinamento interrompido pelo usuário")
        
        # Salvar resultados finais
        if self.best_weights is not None:
            np.save("best_weights_final.npy", self.best_weights)
            np.save("fitness_history.npy", np.array(self.best_fitness_history))
            print(f"\nTreinamento concluído!")
            print(f"Melhor fitness: {self.best_fitness:.2f}")
            print(f"Pesos salvos em 'best_weights_final.npy'")
            
            return self.best_weights, self.best_fitness, self.best_fitness_history
        else:
            print("Nenhum resultado obtido")
            return None, -np.inf, []

def main():
    """Função principal"""
    # Parâmetros de treinamento
    POPULATION_SIZE = 100
    MAX_ITERATIONS = 1000
    TIME_LIMIT_HOURS = 12
    
    trainer = GameTrainer(
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        time_limit_hours=TIME_LIMIT_HOURS
    )
    
    # Executar treinamento
    best_weights, best_fitness, fitness_history = trainer.train(use_parallel=True)
    
    if best_weights is not None:
        print(f"\n=== RESULTADOS FINAIS ===")
        print(f"Melhor fitness alcançado: {best_fitness:.2f}")
        print(f"Total de iterações: {len(fitness_history)}")
        
        # Testar o melhor agente
        print("\n=== TESTANDO MELHOR AGENTE ===")
        test_agent(best_weights, num_tests=5, render=True)

def test_agent(weights: np.ndarray, num_tests: int = 30, render: bool = False):
    """Testa o agente treinado"""
    print(f"Testando agente por {num_tests} execuções...")
    
    scores = []
    agent = NeuralNetworkAgent()
    agent.set_weights(weights)
    
    for i in range(num_tests):
        game_config = GameConfig(render_grid=False)
        game = SurvivalGame(config=game_config, render=render)
        
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            
            if render:
                game.render_frame()
        
        final_score = game.players[0].score
        scores.append(final_score)
        print(f"Teste {i+1}/{num_tests}: Score = {final_score:.2f}")
    
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\nResultados dos testes:")
    print(f"Scores: {scores}")
    print(f"Média: {avg_score:.2f}")
    print(f"Desvio Padrão: {std_score:.2f}")
    
    return scores

if __name__ == "__main__":
    main()