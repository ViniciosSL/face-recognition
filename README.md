# face-recognition
Reconhecimento facial utilizando PCA e kNN

## Aplicação em desenvolvimento!

- [Contextualização](#-contextualização)
- [Técnicas Utilizadas](#-técnicas-utilizadas)
- [Tecnologias](#-tecnologias)
- [Executando a Aplicação](#-executando-a-aplicação)
    - [Docker Container](#-docker-container)
    - [Ambiente Virtual](#-ambiente-virtual)


## Executando a Aplicação

A aplicação pode ser executada por meio de um docker container ou de um ambiente virutal para o python. Abaixo segue um passo a passo para as duas opções.

### Docker Container
Para executar a aplicação em um docker container, basta seguir três etapas:

1. Ative o docker em sua máquina;

2. Abra um novo terminal e navegue até a raíz do repositório clonado;

3. Execute a aplicação.
```shell
> docker-compose run face-recognition
```

### Ambiente Virtual

Como alternativa ao docker, é possível criar um ambiente virtual para o python seguindo os passos:

1. Abra um novo terminal e navegue até a raíz do repositório clonado;

2. Crie e ative um novo ambiente virtual chamado "venv";
```shell
> python -m venv venv
> .\venv\Scripts\activate
```

Caso esteja utilizando o PowerShell lembre-se de ajustar a política de execução de scripts para `RemoteSigned`, assim o shell permite a ativação do ambiente virtual criado
```shell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

3. Instale as dependências por meio do arquivo "requirements.txt";
```shell
> python -m pip install -r requirements.txt
```

4. Execute a aplicação.
```shell
> python .\face_reckon.py
```
