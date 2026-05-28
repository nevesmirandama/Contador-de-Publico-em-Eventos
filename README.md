# Grupo 5 - Contador de Público em Eventos

Projeto inicial em Python + OpenCV para a trilha **Contador de Público em Eventos**.

## Integrantes
- Jairo Galvão
- Adriel Neves

## O que este MVP faz
- abre webcam ou vídeo
- permite definir uma linha virtual por clique e arraste
- detecta objetos em movimento com **MOG2** ou **KNN**
- limpa ruído com operações morfológicas
- cria **bounding boxes** e centróides
- rastreia objetos por ID com associação simples por centróide
- conta cruzamentos em duas direções
- exibe painel com **Entradas**, **Saídas**, **Fluxo/min** e **FPS**
- salva eventos em CSV com timestamp, direção e ID do tracker

## Estrutura
```text
publico_eventos_grupo5/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── main.py
│   ├── pipeline/
│   │   ├── preprocessing.py
│   │   ├── detection.py
│   │   ├── tracking.py
│   │   └── counter.py
│   ├── ui/
│   │   └── line_selector.py
│   └── utils/
│       └── csv_logger.py
├── tests/
│   ├── test_fps.py
│   └── test_failover.py
├── assets/
│   └── samples/
└── results/
```

## Instalação
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
### Webcam
```bash
python src/main.py --source 0
```

### Vídeo gravado
```bash
python src/main.py --source caminho/do/video.mp4
```

### Usando KNN em vez de MOG2
```bash
python src/main.py --source 0 --bg knn
```

### Definindo a linha por parâmetros
```bash
python src/main.py --source 0 --line 100 300 550 300
```

## Controles
- **q**: sair
- **r**: redefinir linha
- **s**: salvar frame atual em `results/`

## Observações
- Na primeira execução sem `--line`, o sistema abre uma tela para você desenhar a linha com o mouse.
- Para a SECOMP, vale muito usar **vídeos próprios de corredores/porta de auditório**, porque o enunciado valoriza isso.
- Este projeto é um **ponto de partida funcional**, não a versão final do trabalho.

## Uso de IA Generativa
- apoio na documentação inicial do README do projeto

Todo o código deve ser lido, entendido e ajustado manualmente pela equipe antes da apresentação.

## Próximos passos sugeridos
1. calibrar `min-area`, `max-distance` e morfologia
2. melhorar o rastreamento para reduzir dupla contagem
3. implementar scripts completos dos 5 protocolos
4. gerar tabelas e gráficos para o artigo
5. preparar pôster e artigo SBC


## Modos de detecção
- **motion**: usa apenas movimento (mais sensível, porém pode contar sombras e partes do corpo)
- **hog**: usa o detector de pessoas do OpenCV
- **hybrid**: combina detector de pessoas + movimento (recomendado)

### Exemplos
```bash
python src/main.py --source 0 --detector hybrid
python src/main.py --source 0 --detector hog
python src/main.py --source video.mp4 --detector hybrid
```
