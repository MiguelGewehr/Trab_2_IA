import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from neural_network import NeuralNetworkAgent
from game.core import SurvivalGame, GameConfig

class StatisticalComparison:
    def __init__(self):
        # Resultados fornecidos no enunciado
        self.rule_based_result = [12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22,
                                 9.95, 19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 
                                 15.13, 22.50, 25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 
                                 15.13, 12.35, 16.19]
        
        self.neural_agent_result = [38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30,
                                   39.76, 48.17, 44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 
                                   67.17, 53.54, 33.59, 49.24, 52.65, 16.35, 44.05, 56.59, 
                                   63.23, 43.96, 43.82, 19.19, 28.36, 18.65]
        
        self.human_result = [27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 
                            20.05, 31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45,
                            12.59, 33.01, 21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 
                            22.96, 9.41, 35.22]
    
    def test_trained_agent(self, weights_file: str, num_tests: int = 30) -> list:
        """
        Testa o agente treinado e retorna os resultados
        """
        print(f"Carregando pesos de: {weights_file}")
        try:
            weights = np.load(weights_file)
        except FileNotFoundError:
            print(f"Arquivo {weights_file} não encontrado!")
            return []
        
        print(f"Testando agente treinado por {num_tests} execuções...")
        
        scores = []
        agent = NeuralNetworkAgent()
        agent.set_weights(weights)
        
        for i in range(num_tests):
            game_config = GameConfig(render_grid=False)
            game = SurvivalGame(config=game_config, render=False)
            
            while not game.all_players_dead():
                state = game.get_state(0, include_internals=True)
                action = agent.predict(state)
                game.update([action])
            
            final_score = game.players[0].score
            scores.append(final_score)
            print(f"Teste {i+1}/{num_tests}: Score = {final_score:.2f}")
        
        return scores
    
    def calculate_statistics(self, results: dict):
        """Calcula estatísticas descritivas para todos os métodos"""
        stats_data = {}
        
        for method_name, scores in results.items():
            stats_data[method_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'scores': scores
            }
        
        return stats_data
    
    def perform_statistical_tests(self, my_results: list):
        """
        Realiza testes estatísticos comparando o método implementado com os outros
        """
        print("\n=== ANÁLISE ESTATÍSTICA ===")
        
        # Dados para comparação
        methods = {
            'Meu Método (Bat + Neural)': my_results,
            'Regras + AG': self.rule_based_result,
            'Neural + AG': self.neural_agent_result,
            'Humano': self.human_result
        }
        
        # Calcular estatísticas descritivas
        stats_data = self.calculate_statistics(methods)
        
        # Imprimir tabela de resultados
        print("\nTabela de Resultados:")
        print("-" * 80)
        print(f"{'Método':<20} {'Média':<10} {'Desvio':<10} {'Min':<10} {'Max':<10}")
        print("-" * 80)
        
        for method, stats in stats_data.items():
            print(f"{method:<20} {stats['mean']:<10.2f} {stats['std']:<10.2f} "
                  f"{stats['min']:<10.2f} {stats['max']:<10.2f}")
        print("-" * 80)
        
        # Imprimir scores individuais
        print("\nScores individuais de cada método:")
        for method, scores in methods.items():
            print(f"\n{method}:")
            scores_formatted = [f"{score:.2f}" for score in scores]
            print(scores_formatted)
        
        # Testes estatísticos
        print("\n=== TESTES ESTATÍSTICOS ===")
        my_method = np.array(my_results)
        
        methods_to_compare = [
            ('Regras + AG', np.array(self.rule_based_result)),
            ('Neural + AG', np.array(self.neural_agent_result)),
            ('Humano', np.array(self.human_result))
        ]
        
        for method_name, other_method in methods_to_compare:
            print(f"\nComparando Meu Método vs {method_name}:")
            
            # Teste t independente
            t_stat, t_pvalue = stats.ttest_ind(my_method, other_method)
            print(f"Teste t independente: t = {t_stat:.4f}, p-value = {t_pvalue:.4f}")
            
            # Teste de Wilcoxon (Mann-Whitney U)
            u_stat, u_pvalue = stats.mannwhitneyu(my_method, other_method, alternative='two-sided')
            print(f"Teste Mann-Whitney U: U = {u_stat:.4f}, p-value = {u_pvalue:.4f}")
            
            # Verificar significância (α = 0.05)
            alpha = 0.05
            t_significant = "SIM" if t_pvalue < alpha else "NÃO"
            u_significant = "SIM" if u_pvalue < alpha else "NÃO"
            
            print(f"Diferença significativa (α=0.05):")
            print(f"  Teste t: {t_significant}")
            print(f"  Teste Mann-Whitney: {u_significant}")
        
        return stats_data
    
    def create_boxplot(self, my_results: list, save_path: str = "comparison_boxplot.png"):
        """
        Cria boxplot comparativo entre os métodos
        """
        # Preparar dados para o boxplot
        all_data = []
        labels = []
        
        methods = {
            'Meu Método\n(Bat + Neural)': my_results,
            'Regras + AG': self.rule_based_result,
            'Neural + AG': self.neural_agent_result,
            'Humano': self.human_result
        }
        
        for method_name, scores in methods.items():
            all_data.extend(scores)
            labels.extend([method_name] * len(scores))
        
        # Criar DataFrame
        df = pd.DataFrame({
            'Score': all_data,
            'Método': labels
        })
        
        # Criar boxplot
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='Método', y='Score', palette='Set2')
        plt.title('Comparação de Desempenho entre Métodos', fontsize=16, fontweight='bold')
        plt.xlabel('Método', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Salvar figura
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nBoxplot salvo em: {save_path}")
        plt.show()
        
        return df
    
    def create_fitness_evolution_plot(self, fitness_history: list, save_path: str = "fitness_evolution.png"):
        """
        Cria gráfico da evolução do fitness durante o treinamento
        """
        plt.figure(figsize=(12, 6))
        iterations = range(1, len(fitness_history) + 1)
        
        plt.plot(iterations, fitness_history, 'b-', linewidth=2, label='Melhor Fitness')
        plt.title('Evolução do Fitness Durante o Treinamento', fontsize=16, fontweight='bold')
        plt.xlabel('Iteração', fontsize=12)
        plt.ylabel('Melhor Fitness', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de evolução salvo em: {save_path}")
        plt.show()
    
    def run_complete_analysis(self, weights_file: str = "best_weights_final.npy", 
                             fitness_history_file: str = "fitness_history.npy"):
        """
        Executa análise completa: teste do agente + comparação estatística + gráficos
        """
        print("=== ANÁLISE COMPLETA DOS RESULTADOS ===")
        
        # 1. Testar agente treinado
        my_results = self.test_trained_agent(weights_file, num_tests=30)
        
        if not my_results:
            print("Não foi possível obter resultados do agente treinado!")
            return
        
        # 2. Análise estatística
        stats_data = self.perform_statistical_tests(my_results)
        
        # 3. Criar boxplot
        df = self.create_boxplot(my_results)
        
        # 4. Gráfico de evolução do fitness (se disponível)
        try:
            fitness_history = np.load(fitness_history_file)
            self.create_fitness_evolution_plot(fitness_history.tolist())
        except FileNotFoundError:
            print(f"Arquivo {fitness_history_file} não encontrado. Pulando gráfico de evolução.")
        
        # 5. Salvar resultados em arquivo
        self.save_results_to_file(my_results, stats_data)
        
        return my_results, stats_data
    
    def save_results_to_file(self, my_results: list, stats_data: dict, 
                           filename: str = "experimental_results.txt"):
        """
        Salva todos os resultados em um arquivo de texto
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== RESULTADOS EXPERIMENTAIS ===\n\n")
            
            # Estatísticas descritivas
            f.write("Estatísticas Descritivas:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Método':<20} {'Média':<10} {'Desvio':<10} {'Min':<10} {'Max':<10}\n")
            f.write("-" * 80 + "\n")
            
            for method, stats in stats_data.items():
                f.write(f"{method:<20} {stats['mean']:<10.2f} {stats['std']:<10.2f} "
                       f"{stats['min']:<10.2f} {stats['max']:<10.2f}\n")
            f.write("-" * 80 + "\n\n")
            
            # Scores individuais
            f.write("Scores Individuais:\n\n")
            
            methods = {
                'Meu Método (Bat + Neural)': my_results,
                'Regras + AG': self.rule_based_result,
                'Neural + AG': self.neural_agent_result,
                'Humano': self.human_result
            }
            
            for method, scores in methods.items():
                f.write(f"{method}:\n")
                scores_str = ", ".join([f"{score:.2f}" for score in scores])
                f.write(f"[{scores_str}]\n\n")
            
            # Testes estatísticos
            f.write("=== TESTES ESTATÍSTICOS ===\n\n")
            my_method = np.array(my_results)
            
            methods_to_compare = [
                ('Regras + AG', np.array(self.rule_based_result)),
                ('Neural + AG', np.array(self.neural_agent_result)),
                ('Humano', np.array(self.human_result))
            ]
            
            for method_name, other_method in methods_to_compare:
                f.write(f"Comparando Meu Método vs {method_name}:\n")
                
                # Teste t independente
                t_stat, t_pvalue = stats.ttest_ind(my_method, other_method)
                f.write(f"Teste t independente: t = {t_stat:.4f}, p-value = {t_pvalue:.4f}\n")
                
                # Teste de Wilcoxon (Mann-Whitney U)
                u_stat, u_pvalue = stats.mannwhitneyu(my_method, other_method, alternative='two-sided')
                f.write(f"Teste Mann-Whitney U: U = {u_stat:.4f}, p-value = {u_pvalue:.4f}\n")
                
                # Verificar significância
                alpha = 0.05
                t_significant = "SIM" if t_pvalue < alpha else "NÃO"
                u_significant = "SIM" if u_pvalue < alpha else "NÃO"
                
                f.write(f"Diferença significativa (α=0.05):\n")
                f.write(f"  Teste t: {t_significant}\n")
                f.write(f"  Teste Mann-Whitney: {u_significant}\n\n")
        
        print(f"Resultados salvos em: {filename}")

def main():
    """Função principal para executar a análise"""
    comparison = StatisticalComparison()
    
    # Executar análise completa
    try:
        my_results, stats_data = comparison.run_complete_analysis()
        print("\nAnálise completa finalizada com sucesso!")
    except Exception as e:
        print(f"Erro durante a análise: {e}")
        print("Certifique-se de que os arquivos de pesos estão disponíveis.")

if __name__ == "__main__":
    main()