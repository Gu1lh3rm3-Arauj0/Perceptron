import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error


def sigmoid(x): 
    #converte entradas para um intervalo entre (0, 1)
    x = np.clip(x, -500, 500)  # prevenção de overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def relu(x): 
    #define negativos como 0, evita gradientes muito pequenos
    return np.clip(np.maximum(0, x), 0, 1000)  # Limita a saída da ReLU para evitar grandes valores

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    #parece o relu, mas permite gradientes pequenos para  negativos
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def tanh(x):
    #intervalo entre (-1, 1), bom pra quando tem positivo e negativo
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

#derivadas servem para o cálculo do gradiente durante o backpropagation

class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.01, momentum=0.9, activation="sigmoid", regularization=0.01):
        #qtd de entrada, qtd de neuronio na camada oculta, qtd de saidas
        #learning_rate (taxa de aprendizado)
            #Valores menores tornam o treinamento mais lento, mas mais estável 
            # valores maiores podem causar instabilidade
        #momentum
            #acelerar o treinamento e evitar que o modelo "pule" de um lado para o outro
            #0.9 significa que 90% da atualização anterior será mantida
        #regularization - controla os valores dos pesos
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization

        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == "leaky_relu":
            self.activation = leaky_relu
            self.activation_derivative = leaky_relu_derivative
        elif activation == "tanh":
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Função de ativação não suportada.")

        #self.weights_hidden = np.random.randn(n_inputs, n_hidden)  # Deve ser (14, 10)
        self.weights_hidden = np.random.randn(n_inputs, n_hidden) * 0.01 #Pesos entre a camada de entrada e a oculta
        self.bias_hidden = np.zeros((1, n_hidden))
        self.weights_output = np.random.randn(n_hidden, n_outputs) * 0.01 #entre oculta e saída
        
        self.bias_output = np.zeros((1, n_outputs))

        self.velocity_hidden = np.zeros_like(self.weights_hidden)#guarda a atualizaçao da oculta
        self.velocity_output = np.zeros_like(self.weights_output)

    def fit(self, X, y, epochs, patience=10): #treinamento
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        best_loss = float('inf') #guarda o melhor erro
        patience_counter = 0 # vai ate o paticence
                             #contas o num de epoch consecutiva sem melhoria

        for epoch in range(epochs):
            hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden #w^T * a + b
            hidden_output = self.activation(hidden_input) #aplica a func ativaçao na entrada calculada

            final_input = np.dot(hidden_output, self.weights_output) + self.bias_output
            final_output = final_input if self.activation == relu else self.activation(final_input)

            error = y - final_output
            #y - ^y

            output_gradient = error
            if self.activation != relu:
                output_gradient *= self.activation_derivative(final_input)
            #O gradiente da camada de saída é a derivada do erro em relação à saída

            hidden_gradient = np.dot(output_gradient, self.weights_output.T) * self.activation_derivative(hidden_input)
            #O gradiente da camada oculta é calculado pela propagação do gradiente da camada de saída

            #regularização 
            reg_hidden = self.regularization * self.weights_hidden
            reg_output = self.regularization * self.weights_output

            #atualizaçao camada de saída
            self.velocity_output = (
                self.momentum * self.velocity_output + self.learning_rate * np.dot(hidden_output.T, output_gradient) - reg_output
            )
            self.weights_output += self.velocity_output #atualiza somando o valor calculado
            self.bias_output += self.learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
                #bias da camada de saida atualiza com base no gradiente
            
            #atualizaçao camada oculta
            self.velocity_hidden = (
                self.momentum * self.velocity_hidden + self.learning_rate * np.dot(X.T, hidden_gradient) - reg_hidden
            )
            self.weights_hidden += self.velocity_hidden
            self.bias_hidden += self.learning_rate * np.sum(hidden_gradient, axis=0, keepdims=True)

            # Verificação de NaN ou Inf
            if np.any(np.isnan(self.weights_hidden)) or np.any(np.isnan(self.weights_output)) or np.any(np.isnan(hidden_output)) or np.any(np.isnan(final_output)):
                print(f"NaN detected in weights or output at epoch {epoch}")
                break
            if np.any(np.isinf(self.weights_hidden)) or np.any(np.isinf(self.weights_output)) or np.any(np.isinf(hidden_output)) or np.any(np.isinf(final_output)):
                print(f"Inf detected in weights or output at epoch {epoch}")
                break

            #calcula a perda atual com o erro quadrático médio
            loss = mean_squared_error(y, final_output)

            #Early Stopping (quando o erro repete mt)
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best loss: {best_loss}")
                break

            #progresso
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss}, Max Weight Hidden {np.max(self.weights_hidden)}, Max Weight Output {np.max(self.weights_output)}")


    def predict(self, X): #previsao das saidas de acorco com o treinamento
        hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden #calculo da entrada da camada oculta
        hidden_output = self.activation(hidden_input)

        final_input = np.dot(hidden_output, self.weights_output) + self.bias_output
        final_output = final_input if self.activation == relu else self.activation(final_input)
        return final_output