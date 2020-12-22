"""
FURB - Universidade Regional de Blumenau
Especialização em Data Science
Aprendizado de Máquina II - Aprendizado Não Supervisionado
Turma 2

> Reconhecimento facial utilizando PCA + kNN
    - Classe FaceRecognizer

Autor: Marcus Moresco Boeno
Último update: 2020-12-22

"""

# Imports de bibliotecas built-in
from math import sqrt

# Import de bibliotecas de terceiros
from matplotlib.image import imread, imsave
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

        # Calcula matriz de covariancia
        cov_mat = np.matmul(diffs.T, diffs)

        # Calcula autovalores e autovetores da matriz de covariancia
        evls, evts = np.linalg.eig(cov_mat)
        
        # Ordena autovalores e autovetores por autovalor decrescente
        evts_sorted = [y for _,y in sorted(zip(evls, evts), reverse=True)]

        # Calcula eigenfaces
        eigenfaces = self.__calcEigenFaces(evts_sorted, diffs, k)

        # # Exporta eigenfaces
        # self.__save_eigen_faces(eigenfaces)

    def __save_eigen_faces(self, eigenfaces:np.array):
        """Exporta k-eigenfaces
        
        > Argumentos:
            - eigenfaces (np.array): Array contendo eigenfaces
        
        > Output:
            - Sem output.
        """
        for pos, eigenface in enumerate(eigenfaces.T):
            imsave(
                "".join(["eigenface_", str(pos+1), ".jpg"]), 
                eigenface.reshape(70, 80).T
                )
    
    def __calcEigenFaces(self, evts:list, diffs:np.array, k:int):
        """Calcula eigenfaces
        
        > Argumentos:
            - evts (list): Lista contendo autovalores e autovetores;
            - diffs (np.arrray): Array contendo imagens diferença;
            - k (int): Número de componentes para cálculo.
        
        > Output:
            - (np.array): Array contendo k-eigenfaces.
        """
        # Monta matriz com autovetores de interesse
        k_evts = evts[0].reshape(evts[0].size, 1)
        for j in range(1, k):
            k_evts = np.hstack((k_evts, evts[j].reshape(evts[j].size, 1)))
        
        # Calcula eigenfaces
        eigenfaces = np.matmul(diffs, k_evts)

        # Aplica normalização L2 em cada eigenface
        eigenfaces = np.apply_along_axis(
            # Função anônima (lambda) para cálculo da normalização L2
            func1d=lambda x: np.divide(x, sqrt(np.sum(np.power(x, 2)))),
            axis=0,
            arr=eigenfaces.T
        ).T

        # Retorna matriz com eigenfaces
        return eigenfaces

    def __calcMean(self) -> np.array:
        """Calcula imagem média

        > Argumentos:
            - Sem argumentos.
        
        > Output:
            - (np.ndarray): Imagem média na forma de vetor coluna.        
        """
        # Le dados da primeira imagem para formar base da imagem média
        mean_img = np.array(imread(self.__imgs[0].data), np.float64)
        
        # Calcula imagem média
        for i in range(1, len(self.__imgs)):
            mean_img += np.array(imread(self.__imgs[i].data), np.float64)
        mean_img = mean_img/len(self.__imgs)

        # Retorna imagem média
        return mean_img.T.reshape(mean_img.size, 1)
    
    def __calcDiff(self, mean_img:np.array) -> np.array:
        """Calcula matriz com diferenças entre imagem média e imagens de
        treino

        > Argumentos:
            - mean_img (np.array): Imagem de treino média
        """
        # Inicia array de diffs com a primeira imagem
        sample = imread(self.__imgs[0].data)
        diffs = np.subtract(sample.T.reshape(sample.size, 1), mean_img)

        # Calcula diferenças entre imagem média e imagens de treino restantes
        for i in range(1, len(self.__imgs)):
            tmp = imread(self.__imgs[i].data)
            tmp = np.subtract(tmp.T.reshape(tmp.size, 1), mean_img)
            diffs = np.hstack((diffs, tmp))
        
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