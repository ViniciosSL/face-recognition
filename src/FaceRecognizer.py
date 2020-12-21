"""
FURB - Universidade Regional de Blumenau
Especialização em Data Science
Aprendizado de Máquina II - Aprendizado Não Supervisionado
Turma 2

> Reconhecimento facial utilizando PCA + kNN
    - Classe FaceRecognizer

Autor: Marcus Moresco Boeno
Último update: 2020-12-20

"""

# Import de bibliotecas de terceiros
from matplotlib.image import imread
import numpy as np

# Imports de bibliotecas da aplicação
from .FaceImage import FaceImage


class FaceRecognizer:
    """Implementação de reconhecimento facial utilizando PCA e kNN

    > Parâmetros de Classe:
        - img_list (list): Lista de imagens da classe FaceImage
    """
    
    # Método construtor
    def __init__(self, img_list):
        self.__imgs = img_list
    
    @property
    def imgs(self) -> list:
        """Método Getter para lista de imgs

        > Arguments:
            - No arguments.
        
        > Output:
            - (list): Lista de objetos da classe FaceImage.
        """
        return self.__imgs_getter
    
    @imgs.setter
    def imgs(self, img_list:str):
        """Método Setter para lista de imgs

        > Arguments:
            - img_list (list): Lista de objetos da classe FaceImage.
        
        > Output:
            - No output.
        """
        self.__imgs_getter = img_list
    
    def fit(self, k:int):
        """Ajusta o modelo de acordo com o número k de componentes

        > Argumentos:
            - k (int): Número de componentes (eigenfaces) a serem
                utilizadas para o treinamento do modelo
        
        > Output:
            - Sem output.
        """
        # Calcula imagem média
        mean_img = self.__calcMean()
        
        # Calcula matriz com diferenças
        diffs = self.__calcDiff(mean_img)

    def __calcMean(self) -> np.array:
        """Calcula imagem média

        > Argumentos:
            - Sem argumentos.
        
        > Output:
            - (np.ndarray): Imagem média na forma de array.        
        """
        # Le dados da primeira imagem para formar base da imagem média
        mean_img = np.array(imread(self.__imgs[0].data), np.float64)
        
        # Calcula imagem média
        for i in range(1, len(self.__imgs)):
            mean_img += np.array(imread(self.__imgs[i].data), np.float64)
        mean_img = mean_img/len(self.__imgs)

        # Retorna imagem média
        return mean_img
    
    def __calcDiff(self, mean_img:np.array) -> np.array:
        """Calcula matriz com diferenças entre imagem média e imagens de
        treino

        > Argumentos:
            - mean_img (np.array): Imagem de treino média
        """
        # Inicia array de diffs com a primeira imagem
        diffs = np.subtract(imread(self.__imgs[0].data), mean_img)

        # Ajusta diffs para array vetor coluna
        diffs = diffs.reshape(diffs.size, 1)

        # Calcula diferenças entre imagem média e imagens de treino restantes
        for i in range(1, len(self.__imgs)):
            tmp_diff = np.subtract(imread(self.__imgs[i].data), mean_img)
            tmp_diff = tmp_diff.reshape(tmp_diff.size, 1)
            diffs = np.hstack((diffs, tmp_diff))
        
        # Retorna array com dados de diferença
        return diffs

    def predict(img) -> tuple:
        """Classifica uma imagem com o modelo treinado

        > Argumentos:
            - img (FaceImage): Imagem da classe FaceImage a ser
                classificada
        
        > Ouput:
            - (tuple): Tupla com 3 elementos:
                [0]: Classificação (label) da imagem
                [1]: Distância ao vizinho mais próximo
                [2]: Erro de reconstrução 
        """
        pass

    def model_accuracy(test_imgs:list) -> float:
        """Acurácia do modelo baseadas em um conjunto de teste

        > Argumentos:
            - test_imgs (list): Lista de objetos da classe FaceImage
                representando o conjunto de teste.
        
        > Output:
            - (float): Acurácia do modelo.
        """
        pass