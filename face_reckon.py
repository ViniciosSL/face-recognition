"""
FURB - Universidade Regional de Blumenau
Especialização em Data Science
Aprendizado de Máquina II - Aprendizado Não Supervisionado
Turma 2

> Reconhecimento facial utilizando PCA + kNN

Autor: Marcus Moresco Boeno
Último update: 2020-12-20

"""

# Imports de bibliotecas built-in
import os
from math import ceil
from random import randint

# Import de bibliotecas de terceiros
from matplotlib.image import imread
import numpy as np

# Imports de bibliotecas da aplicação
from src.FaceRecognizer import FaceRecognizer
from src.FaceImage import FaceImage


def load_dataset(dir_path:str, toTrain:float) -> list:
    """Carrega imagens para reconhecimento facial
    
    As imagens são carregadas e separadas em conjuntos treino e teste 
    utilizando a técnica de Holdout. A divisão é realizada de forma 
    estratificada, levando em consideração o label do indivívuo a imagem
    facial pertence.

    > Argumentos:
        - dir_path (str): Diretório contendo imagens.
        - toTrain (float): Porcentagem de amostras para treino.
    
    > Output:
        - (tuple): Tupla contendo dois elementos:
            - [0]: Lista de amostras de treino;
            - [1]: Lista de amostras de teste
    """

    # Declara dicionário para separação das labels
    face_imgs = {}
    
    # Declara listas para armazenar objetos FaceImage
    train, test = [], []

    # Preenche dicionário
    for folder,_,imgs in os.walk(dir_path):
        for img in imgs:
            label = img.split("_")[1].split(".")[0]
            if label in face_imgs:
                face_imgs[label].append(os.path.join(folder, img))
            else:
                face_imgs[label] = [os.path.join(folder, img)]

    # Separa imagens de treino e teste de forma estratificada
    for label in face_imgs:

        # Número de imagens para treino
        num_imgs_train = ceil(len(face_imgs[label])*toTrain)

        # Separa imagens de treino
        for i in range(num_imgs_train):
            
            # Recupera indice aletorio e caminho relativo da imagem
            randIndex = randint(0, len(face_imgs[label])-1)
            img_path = face_imgs[label].pop(randIndex)
            
            # Cria FaceImage e adiciona no conjunto de treino
            train.append(
                FaceImage(
                    img_id=img_path.split(os.sep)[-1].split("_")[0],
                    label=label,
                    data=img_path
                )
            )

        # Restante das imagens do label vão para conjunto de teste
        for test_img in face_imgs[label]:

            # Cria FaceImage e adiciona no conjunto de teste
            test.append(
                FaceImage(
                    img_id=test_img.split(os.sep)[-1].split("_")[0],
                    label=label,
                    data=test_img
                )
            )
    
    # Retorna tupla com imagens de treino e teste
    return train, test


def main():
    """Aplica classes e métodos para reconhecimento facial

    > Argumentos:
        - Sem argumentos.
    
    > Output:
        - Sem output.
    """
    # Carrega banco de dados separando em treino e teste
    train_imgs, test_imgs = load_dataset("imgs", 0.7)

    # Cria instância da classe Face_Reckognizer
    model = FaceRecognizer(train_imgs)

    # # Realiza treinamento do modelo indicando o número de eigenfaces
    # model.fit(10)

    # # Resgata métricas do modelo utilizando o conjunto de teste
    # accuracy = model.model_accuracy(test_imgs)

    # # Apresenta resumo das métricas em tela
    # print(f"Acurácia de {accuracy.2f}")


if __name__ == "__main__":
    main()