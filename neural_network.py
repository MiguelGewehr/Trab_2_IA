import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=27, hidden_layers=[32, 16], output_size=3):
        """
        Rede Neural para o agente do jogo
        input_size: tamanho da entrada (25 da grade + 2 variáveis internas)
        hidden_layers: lista com tamanho das camadas ocultas
        output_size: 3 ações possíveis (0, 1, 2)
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.layers = [input_size] + hidden_layers + [output_size]
        
        # Inicializar estrutura de pesos e bias
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layers) - 1):
            # Xavier initialization
            w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros(self.layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def set_weights(self, weights_vector):
        """Define os pesos da rede a partir de um vetor (usado pela metaheurística)"""
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            b_size = self.biases[i].size
            
            # Extrair pesos da matriz
            self.weights[i] = weights_vector[idx:idx+w_size].reshape(self.weights[i].shape)
            idx += w_size
            
            # Extrair bias
            self.biases[i] = weights_vector[idx:idx+b_size]
            idx += b_size
    
    def get_weights_size(self):
        """Retorna o tamanho total do vetor de pesos"""
        total_size = 0
        for i in range(len(self.weights)):
            total_size += self.weights[i].size + self.biases[i].size
        return total_size
    
    def tanh(self, x):
        """Função de ativação tangente hiperbólica"""
        return np.tanh(x)
    
    def softmax(self, x):
        """Função de ativação softmax para a camada de saída"""
        exp_x = np.exp(x - np.max(x))  # Para estabilidade numérica
        return exp_x / np.sum(exp_x)
    
    def forward(self, x):
        """Propagação direta pela rede"""
        current_input = x
        
        # Camadas ocultas com tanh
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = self.tanh(z)
        
        # Camada de saída com softmax
        z = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self.softmax(z)
        
        return output
    
    def predict(self, x):
        """Predição da ação (retorna índice da ação)"""
        probabilities = self.forward(x)
        return np.argmax(probabilities)

# Agente que usa a rede neural
class NeuralNetworkAgent:
    def __init__(self, input_size=27, hidden_layers=[32, 16], output_size=3):
        self.network = NeuralNetwork(input_size, hidden_layers, output_size)
    
    def set_weights(self, weights):
        """Define os pesos da rede neural"""
        self.network.set_weights(weights)
    
    def predict(self, state):
        """Prediz a ação baseada no estado atual"""
        return self.network.predict(state)
    
    def get_weights_size(self):
        """Retorna o tamanho do vetor de pesos"""
        return self.network.get_weights_size()