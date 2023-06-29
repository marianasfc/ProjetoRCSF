import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def conversor(valor, unidade1, unidade2):
    if(unidade1 == "dBm" and unidade2 == "dBm"):
        return valor + "dBm = " + valor + "dBm"
    if(unidade1 == "dBm" and unidade2 == "W"):
        return valor + "dBm = " + convertedBmW(valor) + "W"
    if(unidade1 == "dBm" and unidade2 == "dBW"):
        return valor + "dBm = " + convertedBmdBW(valor) + "dBW"
    if(unidade1 == "dBm" and unidade2 == "mW"):
        return valor + "dBm = " + convertedBmmW(valor) + "mW"

    if(unidade1 == "W" and unidade2 == "dBm"):
        return valor + "W = " + converteWdBm(valor) + "dBm"
    if(unidade1 == "W" and unidade2 == "W"):
        return valor + "W = " + valor + "W"
    if(unidade1 == "W" and unidade2 == "dBW"):
        return valor + "W = " + converteWdBW(valor) + "dBW"
    if(unidade1 == "W" and unidade2 == "mW"):
        return valor + "W = " + converteWmW(valor) + "mW"

    if(unidade1 == "dBW" and unidade2 == "dBm"):
        return valor + "dBW = " + convertedBWdBm(valor) + "dBm"
    if(unidade1 == "dBW" and unidade2 == "W"):
        return valor + "dBW = " + convertedBWW(valor) + "W"
    if(unidade1 == "dBW" and unidade2 == "dBW"):
        return valor + "dBW = " + valor + "dBW"
    if(unidade1 == "dBW" and unidade2 == "mW"):
        return valor + "dBW = " + convertedBWmW(valor) + "mW"

    if(unidade1 == "mW" and unidade2 == "dBm"):
        return valor + "mW = " + convertemWdBm(valor) + "dBm"
    if(unidade1 == "mW" and unidade2 == "W"):
        return valor + "dBW = " + convertemWW(valor) + "W"
    if(unidade1 == "mW" and unidade2 == "dBW"):
        return valor + "mW = " + convertemWdBW(valor) + "dBW"
    if(unidade1 == "mW" and unidade2 == "mW"):
        return valor + "mW = " + valor + "mW"


def convertedBmdBW(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = valorOriginal-30
    string = str(valorConvertido)
    return string

def convertedBmW(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = 10**((valorOriginal-30)/10)
    string = str(valorConvertido)
    return string

def convertedBmmW(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = 10**(valorOriginal/10)
    string = str(valorConvertido)
    return string

def comvertedBmdBw(original):
    valorOriginal = float(original)
    valorConvertido = 0
    
    valorConvertido = valorOriginal + 30   
    string = str(valorConvertido)
    return string

def converteWdBm(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = (10*math.log10(valorOriginal / 1)) + 30
    string = str(valorConvertido)
    return string

def converteWmW(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = valorOriginal * 1000
    string = str(valorConvertido)
    return string

def converteWdBW(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = (10*math.log10(valorOriginal / 1))
    string = str(valorConvertido)
    return string

def convertedBWW(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = 10**(valorOriginal/10)
    string = str(valorConvertido)
    return string

def convertedBWdBm(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = valorOriginal+30
    string = str(valorConvertido)
    return string

def convertedBWmW(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = (10**(valorOriginal/10)) * 1000
    string = str(valorConvertido)
    return string

def convertemWdBm(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = (10*math.log10(valorOriginal / 1))
    string = str(valorConvertido)
    return string

def convertemWW(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = valorOriginal / 1000
    string = str(valorConvertido)
    return string

def convertemWdBW(original):
    valorOriginal = float(original)
    valorConvertido = 0

    valorConvertido = (10*math.log10((valorOriginal/1000) / 1))
    string = str(valorConvertido)
    return string


def calculateNyquist(b, v):
    b = int (b)
    v = int (v)
    b = b * (10**6)  # convertendo MHz para Hz
    resultado = b * v * math.log2(v)
    string = str(resultado)
    return "Debito:" + string

def calculateShannon(snr, b):
    b = int (b)
    v = int (v)
    b = b * (10**6)  # convertendo MHz para Hz
    snr = 10 ** (snr / 10)  # convertendo SNR de dB para escala linear
    resultado = b * math.log2(1 + snr)
    string = str(resultado)
    return "Capacidade:" + string


def qam(b, db, ber):
    best_modulation = None
    max_rate = 0
    db = 10 **(db / 10)
    for M in [4, 8, 16, 32, 64, 128, 256]:
        k = int (k)
        pb = int (pb)
        k = math.log2(M)
        pb = 4*k*(1 - 1/math.sqrt(M))*math.erfc(math.sqrt(3*k**2*(M-1)*db/2))
        rate = b * k * math.log2(1 + (1/pb))
        if pb <= ber and rate > max_rate:
            best_modulation = M
            max_rate = rate
    return "A melhor modulação é QAM-" + str(best_modulation) + "com uma taxa máxima de" + str(max_rate) + " bps"


def atenuacao(frequencia, dmax, ptx):
    if dmax == 0:
        return 0
    return 32 + 20 * math.log(int(frequencia)) + 20 * math.log(int(dmax))

def espaco_livre(potencia_tx, frequencia, dmax, sensibilidade):
    if os.path.exists('static/core/grafico1.png'):
        os.remove('static/core/grafico1.png')
    distancias = []
    potencias_rx = []
    potencia_rx_min = int(sensibilidade)
    for a in range(1, int(dmax)):
        distancias.append(a)
    potencias_rx_min = [potencia_rx_min for d in distancias]
    for i in distancias:
        L = atenuacao(frequencia,i, potencia_tx)
        potencias_rx.append(int(potencia_tx) - int(L))
    for d,p in zip(distancias, potencias_rx):
        if p <= potencia_rx_min:
            pmin = p
            dmax = d
            break
    plt.plot(distancias, potencias_rx)
    plt.plot(distancias, potencias_rx_min)
    plt.scatter(dmax, pmin)
    plt.ylabel("Potência Recebida")
    plt.xlabel("Distância")
    plt.title("Atenuação Espaço Livre")
    figura = plt.gcf()
    plt.close()
    figura.savefig('static/core/grafico1.png')


x_area = 100
y_area = 100
d_pixel = 1
i_centro = 50
j_centro = 50

def cria_matriz(x_area, y_area, d_pixel):
    n_x = x_area // d_pixel
    n_y = y_area // d_pixel
    matriz = [[None for y in range(n_y) ] for x in range(n_x)]
    return matriz
        
def calcula_distancia(x_centro, y_centro, x_a, y_a):
    distancia = ((x_centro - x_a)**2 + (y_centro - y_a) **2) ** (1/2)
    return distancia


def calcula_coordenadas_i_j(x_ponto, y_ponto, d_pixel) -> "(i_centro, j_centro)":
    return x_ponto//d_pixel, y_ponto//d_pixel

def calcula_matriz_espaco_livre (d,f,ptx):
    if int(d) == 0:
        d = 1  
    atenuacao = 32 + 20 * math.log(int(f)) + 20 * math.log(int(d))        
    prx = int(ptx)  - atenuacao
    return prx

def calcula_matriz_prx(matriz, ptx, f, calcula_matriz_espaco_livre, i_centro, j_centro, d_pixel):
    for y in range(len(matriz)):
        for x in range(len(matriz[0])):
            d = calcula_distancia(i_centro, j_centro, x, y)*d_pixel
            matriz[x][y] = calcula_matriz_espaco_livre(d,int(f),int(ptx))      
    return matriz

def grafico(matriz_ptx): 
    if os.path.exists('static/core/grafico2.png'):
        os.remove('static/core/grafico2.png')
    sns.heatmap(matriz_ptx)
    plt.autoscale()
    figura = plt.gcf()
    plt.close()
    figura.savefig('static/core/grafico2.png')
    

def dadosPlo():
    if os.path.exists('static/core/grafico4.png'):
        os.remove('static/core/grafico4.png')
    df = pd.read_csv('static/core/campanha_de_medidas.txt', sep=',', header=None, names=['Distância (m)', 'Potência Recebida (dBm)'])
    # Extraindo as colunas do DataFrame
    distancias = df['Distância (m)']
    potencias = df['Potência Recebida (dBm)']
    # Ajuste da regressão polinomial
    coefficients = np.polyfit(distancias, potencias, 2)  # Grau do polinômio: 2
    polynomial = np.poly1d(coefficients)
    # Valores preditos pela regressão polinomial
    potencias_preditas = polynomial(distancias)
    # Visualização dos dados e da regressão polinomial
    plt.scatter(distancias, potencias, label='Medições')
    plt.plot(distancias, potencias_preditas, color='red', label='Regressão Polinomial')
    plt.xlabel('Distância (m)')
    plt.ylabel('Potência Recebida (dBm)')
    plt.title('Decaimento do Sinal WiFi com a Distância')
    plt.legend()
    plt.grid(True)
    figura = plt.gcf()
    plt.close()
    figura.savefig('static/core/grafico4.png')
    # Cálculo do R² (R-value)
    ss_total = np.sum((potencias - np.mean(potencias)) ** 2)
    ss_residual = np.sum((potencias - potencias_preditas) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    print('O valor do R² é:', r_squared)
    

def diagrama_polar(angulos, ganhos, cor):
    """
    Lista de ângulos deve estar em radianos
    Lista de ganhos correspondentes, que deverão ser negativos (pois são perdas)
    """
    ax = plt.subplot(projection='polar')
    ax.plot(angulos, ganhos)
    ax.set_rmin(-40)  # Ganho mínimo representado
    ax.set_thetagrids([r for r in range(0, 360, 15)])  # Grelha angular
    ax.set_rgrids([r for r in range(0, -40, -3)])  # Grelha radial
    ax.set_theta_offset(math.pi/2)  # Offset para a localização de 0 em radianos
    ax.set_theta_direction(-1)  # Direção na qual os ângulos crescem
    ax.set_rlabel_position(0)  # Afastar etiquetas radiais da linha desenhada
    ax.grid(True)
    ax.set_title("Diagrama de Radiação", fontsize=14, pad=20)  # Distância do título ao gráfico


def calcular_potencia_recebida(distancia, ganho_antena, ptx):
    # Cálculos baseados no modelo de propagação
    # Exemplo usando o modelo de propagação Friis
    potencia_recebida = 10 * np.log10(ptx) + ganho_antena - 20 * np.log10(distancia)
    return potencia_recebida

def fator_antena(angulo_rad, azimute_rad):
    fator = math.cos(int(angulo_rad)) * math.cos(int(azimute_rad))
    return fator

def obter_diagrama_radiacao_antena(antena, azimute_antena):
    if antena == 'Antena Padrão':
        ganhos = {
            0: 10.0,
            90: 8.0,
            180: 5.0
        }
    elif antena == 'Antena Especial':
        ganhos = {
            0: 12.0,
            90: 10.0,
            180: 7.0
        }
    else:
        ganhos = {}  
    return ganhos.get(azimute_antena, 0.0)

def calcular_heatmap_potencia(ptx, f, modelo_propagacao, largura=1000, altura=1000, tamanho_pixel=10,
                              antena='Antena Padrão', azimute_antena=0, altura_antena=20, altura_terminal=1.5):
    # Cálculo do número de pixels
    potencia_recebida = 0
    num_pixels_largura = int(largura) / int(tamanho_pixel)
    num_pixels_altura = int(altura) / int(tamanho_pixel)
    # Criação da matriz para o heatmap da potência recebida
    heatmap = np.zeros((int(num_pixels_altura), int(num_pixels_largura)))
    # Cálculo do diagrama de radiação da antena
    diagrama_radiacao = obter_diagrama_radiacao_antena(antena, azimute_antena)
    # Loop para calcular a potência recebida em cada pixel
    for i in range(int(num_pixels_altura)):
        for j in range(int(num_pixels_largura)):
            # Cálculo da distância do pixel à antena
            distancia = np.sqrt((i * int(tamanho_pixel))**2 + (j * int(tamanho_pixel))**2)
            # Cálculo da potência recebida no pixel
            if modelo_propagacao == 'Apenas Horizontal':
                # Cálculo considerando apenas o diagrama de radiação horizontal
                potencia_recebida = int(ptx) / (4 * np.pi * distancia)**2
            elif modelo_propagacao == 'Horizontal e Vertical':
                # Cálculo considerando o diagrama de radiação horizontal e vertical
                angulo_rad = np.arctan2(int(altura_terminal) - int(altura_terminal), distancia)
                azimute_rad = np.arctan2(j * int(tamanho_pixel), i * int(tamanho_pixel))
                potencia_recebida = int(ptx) / (4 * np.pi * distancia)**2 * fator_antena(angulo_rad, azimute_rad)
            # Armazenamento da potência recebida no heatmap
            heatmap[i, j] = potencia_recebida
            heatmap = np.reshape(heatmap, (int(num_pixels_altura), int(num_pixels_largura)))
    # Plotar o heatmap da potência recebida
    plt.imshow(heatmap, cmap='jet', extent=[0, int(largura), 0, int(altura)])
    plt.colorbar(label='Potência Recebida (dBm)')
    plt.xlabel('Distância (m)')
    plt.ylabel('Distância (m)')
    plt.title('Heatmap da Potência Recebida no Cenário')
    figura = plt.gcf()
    plt.close()
    figura.savefig('static/core/grafico3.png')
                          
def plot_variacao_atenuacao_freq(frequencias, constante_atenuacao):
    constante_atenuacao1 = 0
    # Converter as frequências para uma lista de valores numéricos
    frequencias = [int(f) for f in frequencias.split(",")]
    # Definir a distância
    distancia = np.linspace(1, 1000, 100)  # Exemplo: distância varia de 1 a 1000 metros
    # Plotar as curvas de variação da atenuação
    for freq in frequencias:
        if constante_atenuacao == 'espaço livre':
            constante_atenuacao1 = 2
        elif constante_atenuacao == 'ambiente urbano':
            constante_atenuacao1 = 3.3   
        elif constante_atenuacao == 'indoor':
            constante_atenuacao1 = 4
        atenuacao = 20 * np.log10(distancia) + 20 * np.log10(freq) + constante_atenuacao1
        plt.plot(distancia, atenuacao, label=f"{freq} MHz")
    # Configurar o gráfico
    plt.xlabel("Distância (metros)")
    plt.ylabel("Atenuação (dB)")
    plt.legend()
    plt.grid(True)
    figura = plt.gcf()
    plt.close()
    figura.savefig('static/core/grafico5.png')

def plot_variacao_atenuacao(frequencia):
    atenuacoes = [2, 3.3, 4]
    atenuacoes_name = ['espaço livre', 'ambiente urbano', 'indoor']
    distancia = np.linspace(1, 1000, 100)  # Exemplo: distância varia de 1 a 1000 metros
    for i in range(len(atenuacoes)):
        atenuacao = 20 * np.log10(distancia) + 20 * np.log10(int(frequencia)) + atenuacoes[i]
        plt.plot(distancia, atenuacao, label=f"n = {atenuacoes_name[i]}")
    plt.xlabel("Distância (metros)")
    plt.ylabel("Atenuação (dB)")
    plt.legend()
    plt.grid(True)
    figura = plt.gcf()
    plt.close()
    figura.savefig('static/core/grafico8.png')


def plot_relacao_ci():
    if os.path.exists('static/core/grafico6.png'):
        os.remove('static/core/grafico6.png')
    # Dados fictícios para as relações C/I para diferentes padrões celulares
    padroes_celulares = ['GSM', 'UMTS', 'LTE', '5G']
    relacao_ci = [10, 15, 20, 25]
    # Plotar o gráfico de barras das relações C/I
    plt.bar(padroes_celulares, relacao_ci)
    plt.xlabel('Padrão Celular')
    plt.ylabel('Relação C/I (dB)')
    plt.title('Relação C/I para Diferentes Padrões Celulares')
    figura = plt.gcf()
    plt.close()
    figura.savefig('static/core/grafico6.png')

def plot_heatmap_ci():
    if os.path.exists('static/core/grafico7.png'):
        os.remove('static/core/grafico7.png')
    # Dados fictícios para o C/I
    padroes_celulares = ['GSM', 'UMTS', 'LTE', '5G']
    matriz_ci = np.random.rand(7, 7) * 30  # Matriz 7x7 com valores aleatórios de C/I
    # Configurações do heatmap
    cmap = 'coolwarm'  # Escolha um mapa de cores adequado
    center_cell_color = 'white'  # Cor da célula "útil" no centro
    annot = True  # Exibir valores na célula
    # Plotar o heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_ci, cmap=cmap, annot=annot, center=matriz_ci[3, 3], cbar=True)
    plt.title('Heatmap de C/I para Padrões Celulares')
    plt.xticks(np.arange(7) + 0.5, np.arange(7))
    plt.yticks(np.arange(7) + 0.5, np.arange(7))
    plt.gca().invert_yaxis()  # Inverter a ordem dos eixos y
    plt.gca().set_xticklabels([])  # Ocultar rótulos do eixo x
    plt.gca().set_yticklabels([])  # Ocultar rótulos do eixo y
    plt.gca().add_patch(plt.Rectangle((2.5, 2.5), 2, 2, fill=False, edgecolor=center_cell_color, linewidth=2))  # Desenhar retângulo para célula "útil"
    figura = plt.gcf()
    plt.close()
    figura.savefig('static/core/grafico7.png')

def erlang_b_probabilidade_bloqueio(tráfego, numero_canais):
    soma = 0
    for k in range(int(numero_canais) + 1):
        soma += (int(tráfego) ** k) / math.factorial(k)
    prob_bloqueio = ((int(tráfego) ** int(numero_canais)) / math.factorial(int(numero_canais))) / soma
    return prob_bloqueio

def erlang_b_numero_canais(trafego, prob_bloqueio):
    numero_canais = 1
    while True:
        prob_bloqueio_calculado = 0
        for n in range(numero_canais + 1):
            prob_bloqueio_calculado += (int(trafego) ** n) / (math.factorial(n))
        prob_bloqueio_calculado = ((int(trafego) ** numero_canais) / (math.factorial(numero_canais))) / prob_bloqueio_calculado
        if prob_bloqueio_calculado <= int(prob_bloqueio):
            break
        numero_canais += 1
    return numero_canais

def erlang_b_tráfego(taxa_chegada, duração_média):
    tráfego = int(taxa_chegada) * int(duração_média)
    return tráfego
