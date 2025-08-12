#!/usr/bin/env python3
"""
Script principal para executar todo o experimento do trabalho
Matrícula terminada em 5: Rede Neural + Bat Algorithm
"""

import numpy as np
import sys
import os
import time
import argparse
from pathlib import Path

# Adicionar paths necessários
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_training import GameTrainer
from statistical_comparison import StatisticalComparison

def train_agent(population_size=100, max_iterations=1000, time_limit_hours=12, 
                use_parallel=True, save_prefix="experiment"):
    """
    Fase 1: Treinar o agente usando Bat Algorithm + Neural Network
    """
    print("=" * 60)
    print("FASE 1: TREINAMENTO DO AGENTE")
    print("=" * 60)
    print(f"Algoritmo: Bat Algorithm")
    print(f"Classificador: Rede Neural")
    print(f"População: {population_size}")
    print(f"Iterações máximas: {max_iterations}")
    print(f"Limite de tempo: {time_limit_hours} horas")
    print(f"Modo paralelo: {use_parallel}")
    print()
    
    # Criar trainer
    trainer = GameTrainer(
        population_size=population_size,
        max_iterations=max_iterations,
        time_limit_hours=time_limit_hours
    )
    
    # Executar treinamento
    start_time = time.time()
    best_weights, best_fitness, fitness_history = trainer.train(use_parallel=use_parallel)
    training_time = time.time() - start_time
    
    # Salvar resultados
    if best_weights is not None:
        weights_file = f"{save_prefix}_best_weights.npy"
        history_file = f"{save_prefix}_fitness_history.npy"
        
        np.save(weights_file, best_weights)
        np.save(history_file, np.array(fitness_history))
        
        print(f"\nTREINAMENTO CONCLUÍDO!")
        print(f"Tempo total: {training_time/3600:.2f} horas")
        print(f"Melhor fitness: {best_fitness:.2f}")
        print(f"Iterações executadas: {len(fitness_history)}")
        print(f"Pesos salvos em: {weights_file}")
        print(f"Histórico salvo em: {history_file}")
        
        return weights_file, history_file, best_fitness
    else:
        print("ERRO: Treinamento falhou!")
        return None, None, -np.inf

def analyze_results(weights_file, history_file):
    """
    Fase 2: Análise dos resultados e comparação estatística
    """
    print("\n" + "=" * 60)
    print("FASE 2: ANÁLISE DOS RESULTADOS")
    print("=" * 60)
    
    # Criar objeto de comparação
    comparison = StatisticalComparison()
    
    # Executar análise completa
    try:
        my_results, stats_data = comparison.run_complete_analysis(
            weights_file=weights_file,
            fitness_history_file=history_file
        )
        
        print("\nANÁLISE CONCLUÍDA!")
        print("Arquivos gerados:")
        print("- comparison_boxplot.png")
        print("- fitness_evolution.png")
        print("- experimental_results.txt")
        
        return True
    
    except Exception as e:
        print(f"ERRO na análise: {e}")
        return False

def create_summary_report(weights_file, history_file, best_fitness, training_time=None):
    """
    Fase 3: Criar relatório resumo para o artigo
    """
    print("\n" + "=" * 60)
    print("FASE 3: RELATÓRIO RESUMO")
    print("=" * 60)
    
    try:
        # Carregar dados
        fitness_history = np.load(history_file)
        
        # Calcular estatísticas do treinamento
        final_fitness = fitness_history[-1]
        max_fitness = np.max(fitness_history)
        iterations_executed = len(fitness_history)
        
        # Carregar resultados experimentais se existirem
        comparison = StatisticalComparison()
        my_results = comparison.test_trained_agent(weights_file, num_tests=30)
        
        # Criar relatório
        report_content = f"""
=== RELATÓRIO RESUMO DO EXPERIMENTO ===

CONFIGURAÇÃO:
- Matrícula terminada em: 5
- Metaheurística: Bat Algorithm (Voo dos Morcegos)
- Classificador: Rede Neural (32-16-3 neurônios, tanh + softmax)
- População: 100 indivíduos
- Limite de tempo: 12 horas

RESULTADOS DO TREINAMENTO:
- Melhor fitness alcançado: {best_fitness:.2f}
- Fitness final: {final_fitness:.2f}
- Iterações executadas: {iterations_executed}
- Tempo de treinamento: {training_time/3600:.2f} horas (estimado)

ARQUITETURA DA REDE NEURAL:
- Entrada: 27 neurônios (25 da grade 5x5 + 2 variáveis internas)
- Camada oculta 1: 32 neurônios (ativação: tanh)
- Camada oculta 2: 16 neurônios (ativação: tanh)
- Camada de saída: 3 neurônios (ativação: softmax)
- Total de pesos otimizados: {len(np.load(weights_file))}

DESEMPENHO NOS TESTES (30 execuções):
- Média: {np.mean(my_results):.2f}
- Desvio padrão: {np.std(my_results):.2f}
- Mínimo: {np.min(my_results):.2f}
- Máximo: {np.max(my_results):.2f}

COMPARAÇÃO COM OUTROS MÉTODOS:
Ver arquivo 'experimental_results.txt' para análise estatística completa.

ARQUIVOS GERADOS:
- {weights_file}: Melhores pesos da rede neural
- {history_file}: Evolução do fitness durante treinamento
- comparison_boxplot.png: Boxplot comparativo
- fitness_evolution.png: Evolução do fitness
- experimental_results.txt: Análise estatística completa
- summary_report.txt: Este relatório

=== FIM DO RELATÓRIO ===
        """
        
        # Salvar relatório
        with open("summary_report.txt", "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print("Relatório resumo salvo em: summary_report.txt")
        print("\nRESUMO EXECUTIVO:")
        print(f"- Melhor fitness: {best_fitness:.2f}")
        print(f"- Iterações: {iterations_executed}")
        print(f"- Desempenho médio nos testes: {np.mean(my_results):.2f} ± {np.std(my_results):.2f}")
        
        return True
        
    except Exception as e:
        print(f"ERRO ao criar relatório: {e}")
        return False

def main():
    """Função principal que executa todo o experimento"""
    parser = argparse.ArgumentParser(description='Experimento Completo - Bat Algorithm + Neural Network')
    parser.add_argument('--population', type=int, default=100, help='Tamanho da população')
    parser.add_argument('--iterations', type=int, default=1000, help='Número máximo de iterações')
    parser.add_argument('--time-limit', type=float, default=12.0, help='Limite de tempo em horas')
    parser.add_argument('--no-parallel', action='store_true', help='Desabilitar processamento paralelo')
    parser.add_argument('--skip-training', action='store_true', help='Pular treinamento e ir direto para análise')
    parser.add_argument('--weights-file', type=str, default='experiment_best_weights.npy', help='Arquivo de pesos')
    parser.add_argument('--history-file', type=str, default='experiment_fitness_history.npy', help='Arquivo de histórico')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXPERIMENTO: APRENDIZADO POR REFORÇO - SPACE INVADERS")
    print("MATRÍCULA TERMINADA EM 5")
    print("METAHEURÍSTICA: BAT ALGORITHM")
    print("CLASSIFICADOR: REDE NEURAL")
    print("=" * 80)
    
    start_experiment = time.time()
    
    # Fase 1: Treinamento (se não for pulado)
    if not args.skip_training:
        weights_file, history_file, best_fitness = train_agent(
            population_size=args.population,
            max_iterations=args.iterations,
            time_limit_hours=args.time_limit,
            use_parallel=not args.no_parallel,
            save_prefix="experiment"
        )
        
        if weights_file is None:
            print("EXPERIMENTO FALHOU NO TREINAMENTO!")
            return
    else:
        weights_file = args.weights_file
        history_file = args.history_file
        best_fitness = 0  # Será calculado na análise
        print("PULANDO TREINAMENTO - usando arquivos existentes")
    
    # Fase 2: Análise dos resultados
    analysis_success = analyze_results(weights_file, history_file)
    
    if not analysis_success:
        print("EXPERIMENTO FALHOU NA ANÁLISE!")
        return
    
    # Fase 3: Relatório final
    training_time = time.time() - start_experiment
    report_success = create_summary_report(weights_file, history_file, best_fitness, training_time)
    
    if not report_success:
        print("EXPERIMENTO FALHOU NO RELATÓRIO!")
        return
    
    # Conclusão
    total_time = time.time() - start_experiment
    print("\n" + "=" * 80)
    print("EXPERIMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print(f"Tempo total: {total_time/3600:.2f} horas")
    print(f"Arquivos gerados na pasta atual:")
    print(f"- {weights_file}")
    print(f"- {history_file}")
    print("- comparison_boxplot.png")
    print("- fitness_evolution.png")
    print("- experimental_results.txt")
    print("- summary_report.txt")
    print()
    print("Próximos passos:")
    print("1. Use os dados em 'experimental_results.txt' para escrever o artigo")
    print("2. Inclua os gráficos gerados no artigo")
    print("3. Use 'summary_report.txt' como referência para conclusões")
    print("=" * 80)

if __name__ == "__main__":
    main()