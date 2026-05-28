# Contador de Publico em Eventos

Sistema de contagem automatica de publico em eventos usando visao computacional com Python e OpenCV.

O projeto identifica pessoas ou objetos em movimento em uma cena, acompanha seus centroides ao longo dos frames e contabiliza entradas e saidas quando ha cruzamento de uma linha virtual definida pelo usuario.

## Integrantes

- Jairo Galvao
- Adriel Neves

## Objetivo

O objetivo do sistema e auxiliar a estimativa de fluxo de pessoas em ambientes de eventos, como corredores, portas de auditorios, salas e areas de circulacao.

A aplicacao permite processar imagens de webcam ou videos gravados, exibir a contagem em tempo real e registrar os eventos detectados em arquivo CSV para analise posterior.

## Funcionalidades

- Captura de video por webcam ou arquivo local.
- Definicao de linha virtual por mouse ou por parametros no terminal.
- Deteccao de movimento com subtracao de fundo MOG2 ou KNN.
- Deteccao de pessoas com HOG do OpenCV.
- Modo hibrido combinando deteccao de pessoas e deteccao de movimento.
- Limpeza da mascara com operacoes morfologicas.
- Geracao de bounding boxes e centroides.
- Rastreamento simples por associacao de centroides.
- Contagem de cruzamentos em duas direcoes: entrada e saida.
- Painel em tempo real com entradas, saidas, fluxo por minuto, FPS e detector usado.
- Exportacao dos eventos para CSV com timestamp, direcao, ID do rastreador e posicao do centroide.
- Salvamento manual de frames durante a execucao.

## Como o sistema funciona

1. O video e aberto a partir da webcam ou de um arquivo informado pelo usuario.
2. O usuario define uma linha de contagem na primeira imagem do video.
3. Cada frame passa por uma etapa de deteccao, que pode usar movimento, HOG ou modo hibrido.
4. As deteccoes sao convertidas em caixas delimitadoras e centroides.
5. O rastreador associa as deteccoes atuais aos objetos acompanhados anteriormente.
6. Quando um centroide cruza a linha virtual, o sistema registra entrada ou saida.
7. Os resultados aparecem na tela e tambem sao salvos no arquivo `results/events.csv`.

## Estrutura do projeto

```text
Version_2/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- comando.txt
|-- assets/
|-- results/
|   `-- events.csv
|-- src/
|   |-- main.py
|   |-- pipeline/
|   |   |-- preprocessing.py
|   |   |-- detection.py
|   |   |-- tracking.py
|   |   `-- counter.py
|   |-- ui/
|   |   `-- line_selector.py
|   `-- utils/
|       `-- csv_logger.py
`-- tests/
    |-- test_fps.py
    `-- test_failover.py
```

## Tecnologias utilizadas

- Python 3
- OpenCV
- NumPy
- Subtracao de fundo MOG2/KNN
- Detector HOG para pessoas
- Rastreamento por centroide

## Instalacao

### Windows

```bash
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Como executar

### Usando webcam

```bash
python src/main.py --source 0
```

### Usando video gravado

```bash
python src/main.py --source caminho/do/video.mp4
```

### Usando KNN como subtrator de fundo

```bash
python src/main.py --source 0 --bg knn
```

### Usando o detector HOG

```bash
python src/main.py --source 0 --detector hog
```

### Usando o modo hibrido

```bash
python src/main.py --source 0 --detector hybrid
```

### Definindo a linha por parametros

```bash
python src/main.py --source 0 --line 100 300 550 300
```

## Parametros principais

| Parametro | Descricao | Padrao |
|---|---|---|
| `--source` | Fonte do video. Use `0` para webcam ou informe o caminho de um arquivo. | `0` |
| `--bg` | Algoritmo de subtracao de fundo. Aceita `mog2` ou `knn`. | `mog2` |
| `--detector` | Modo de deteccao. Aceita `motion`, `hog` ou `hybrid`. | `motion` |
| `--min-area` | Area minima para aceitar um objeto detectado por movimento. | `1800` |
| `--max-distance` | Distancia maxima para associar centroides ao mesmo rastreador. | `90.0` |
| `--max-misses` | Numero maximo de frames sem deteccao antes de remover um rastreador. | `25` |
| `--csv` | Caminho do arquivo CSV de eventos. | `results/events.csv` |
| `--line` | Coordenadas da linha fixa no formato `x1 y1 x2 y2`. | manual |
| `--save-frame-dir` | Pasta para salvar frames durante a execucao. | `results` |

## Controles durante a execucao

- `Q`: encerra a aplicacao.
- `R`: redefine a linha de contagem.
- `S`: salva o frame atual na pasta configurada.

## Saida CSV

Os eventos detectados sao salvos em `results/events.csv`.

Formato das colunas:

```csv
timestamp,direction,tracker_id,cx,cy
```

Exemplo:

```csv
2026-05-10T19:40:12,entrada,3,421,250
2026-05-10T19:40:18,saida,4,390,248
```

## Scripts auxiliares

O projeto possui dois scripts auxiliares na pasta `tests/`.

### Medicao de FPS

```bash
python tests/test_fps.py
```

Esse script captura frames da fonte de video e exibe FPS medio, minimo, maximo e desvio-padrao.

### Teste de tolerancia a falhas

```bash
python tests/test_failover.py
```

Esse script auxilia a verificar o comportamento da captura quando ha falha temporaria da webcam.

## Limitacoes

- A precisao depende da iluminacao, posicionamento da camera, oclusoes e calibracao da linha.
- O modo baseado apenas em movimento pode contar sombras ou partes do corpo em alguns cenarios.
- O detector HOG pode ter dificuldade com pessoas pequenas, parcialmente ocultas ou em angulos desfavoraveis.
- O rastreamento por centroide e simples e pode trocar IDs em cenas muito cheias.

## Licenca

Este projeto esta disponibilizado sob a licenca MIT. Consulte o arquivo `LICENSE` para mais detalhes.
