<div align="center">

# Z O L D A X . I A

**MiniGPT construído do zero com PyTorch**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20UI-lightgrey?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## Sobre o projeto

Zoldax.IA é um modelo de linguagem (LLM) implementado **do zero**, sem frameworks de alto nível como Ollama ou LangChain. A arquitetura é baseada no GPT-2 (Transformer), com tokenizer BPE próprio, interface web e sistema de aprendizado supervisionado.

Desenvolvido para fins educacionais e machine learning.

---

## Funcionalidades

- Arquitetura Transformer completa (Multi-Head Attention, Feed Forward, LayerNorm)
- Tokenizer BPE (Byte Pair Encoding) treinado do zero
- Treinamento e retreino contínuo com checkpoint
- Sistema de aprendizado supervisionado (`/ensinar`, `/corrigir`)
- Respostas fixas com prioridade sobre o modelo
- Interface web estilo ChatGPT (Flask) acessível na rede local
- Histórico de conversas em JSON Lines
- Suporte a CPU e CUDA

---

## Estrutura do projeto

```
zoldax-ia/
├── zoldax.py          # CLI — Treino e chat no terminal
├── zoldax_web.py      # Interface web (Flask)
├── texto.txt          # Corpus principal de treino
├── aprendizado.txt    # Pares Q&A aprovados pelo usuário
├── conversas.jsonl    # Histórico de conversas (gerado)
├── zoldax_bpe.pt      # Checkpoint do modelo (gerado)
├── requirements.txt   # Dependências
├── .gitignore
└── README.md
```

---

## Instalação

```bash
git clone https://github.com/seu-usuario/zoldax-ia.git
cd zoldax-ia
```

### Pré-requisitos

```bash
sudo apt update
sudo apt install -y python3-full python3-venv python3-pip
```

### Setup

```bash
python3 -m venv torch-env
source torch-env/bin/activate
pip install --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir numpy flask
```

---

## Como usar

### 1. Ativar o ambiente virtual

```bash
source torch-env/bin/activate
```

### 2. Adicionar conteúdo ao corpus

Edite o `texto.txt` com o conteúdo que o Zoldax vai aprender.
Quanto mais texto, melhores as respostas.

### 3. Treinar o modelo

```bash
# Treinamento inicial
python3 zoldax.py --treino

# Com número de iterações customizado
python3 zoldax.py --treino --steps 5000

# Continuar treinamento anterior
python3 zoldax.py --continue --steps 1000
```

### 4. Usar via terminal

```bash
python3 zoldax.py
```

**Comandos disponíveis no chat:**

| Comando | Descrição |
|---------|-----------|
| `/ensinar` | Cadastra um par pergunta/resposta correto |
| `/corrigir` | Corrige a última resposta gerada |
| `/sair` | Encerra a conversa |

### 5. Usar via interface web

```bash
python3 zoldax_web.py
```

Acesse de qualquer PC na rede:
```
http://<IP-DO-SERVIDOR>:5000
```

Para descobrir seu IP:
```bash
ip a | grep "inet " | grep -v 127
```

---

## Regra de retreino

| Texto adicionado | Comando recomendado |
|---|---|
| Pouco (~1–2 KB) | `--continue --steps 1000` |
| Médio (~10 KB+) | `--treino --steps 3000` |
| Muito (~50 KB+) | `--treino --steps 5000` |

> **Dica:** `--treino` sempre dá resultado mais limpo porque o BPE tokenizer é retreinado junto com o novo vocabulário.

---

## Arquitetura

```
Texto bruto
    ↓
[BPETokenizer]         256 bytes base + fusões aprendidas
    ↓
[Token Embedding]      N_EMBD = 64
[Position Embedding]   BLOCK_SIZE = 64
    ↓
[Block × N_LAYER]
  ├─ MultiHeadAttention (N_HEAD cabeças)
  ├─ FeedForward (ReLU, 4× expansão)
  └─ LayerNorm + Residual Connection
    ↓
[LayerNorm final]
    ↓
[Linear → vocab_size]
    ↓
Texto gerado (temperature + top-k sampling)
```

### Hiperparâmetros padrão

| Parâmetro | Valor |
|-----------|-------|
| `BLOCK_SIZE` | 64 |
| `N_EMBD` | 64 |
| `N_HEAD` | 4 |
| `N_LAYER` | 2 |
| `DROPOUT` | 0.1 |
| `LEARNING_RATE` | 3e-4 |
| `MAX_MERGES` (BPE) | 32 |
| `TEMPERATURE` | 0.8 |
| `TOP_K` | 20 |

---

## Interface Web

<div align="center">

> Interface escura estilo ChatGPT, acessível na rede local via Flask.

</div>

**Funcionalidades da interface:**
- Chat com animação de digitação
- Sidebar com status do dispositivo (CPU/CUDA)
- Modal "Ensinar resposta" para cadastrar Q&A pelo browser
- Nova conversa
- Histórico salvo automaticamente

---

## Requisitos de hardware

| RAM disponível | Configuração recomendada |
|---|---|
| 4 GB | `N_LAYER=2`, `N_EMBD=64`, corpus até 50 KB |
| 8 GB | `N_LAYER=4`, `N_EMBD=128`, corpus até 500 KB |
| 16 GB+ | `N_LAYER=6`, `N_EMBD=256`, corpus maior |

---

## Licença

MIT License — veja [LICENSE](LICENSE).

---

<div align="center">

Desenvolvido by **Saraiva Digital**

</div>
