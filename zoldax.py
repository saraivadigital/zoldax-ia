import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# ZOLDAX.IA 1.2 - by Saraiva Digital
# Memoria + aprendizado supervisionado + treinamento continuo
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEXT_FILE       = os.path.join(BASE_DIR, "texto.txt")
LEARNING_FILE   = os.path.join(BASE_DIR, "aprendizado.txt")
HISTORY_FILE    = os.path.join(BASE_DIR, "conversas.jsonl")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "zoldax_bpe.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Arquitetura
BATCH_SIZE = 8
BLOCK_SIZE = 64
N_EMBD     = 64
N_HEAD     = 4
N_LAYER    = 2
DROPOUT    = 0.1

# Treinamento
LEARNING_RATE  = 3e-4
MAX_MERGES     = 32
DEFAULT_STEPS  = 3000
EVAL_INTERVAL  = 200
EVAL_ITERS     = 20

# Conversa
MAX_NEW_TOKENS = 100
TEMPERATURE    = 0.8
TOP_K          = 20

torch.manual_seed(1337)


# ============================================================
# TOKENIZADOR BPE
# ============================================================

class BPETokenizer:

    def __init__(self):
        self.merges = []
        self.vocab  = {}

    def train(self, text, max_merges=MAX_MERGES):

        tokens     = list(text.encode("utf-8"))
        self.merges = []

        print("\nAprendendo vocabulário BPE...")

        for i in range(max_merges):

            if len(tokens) < 2:
                break

            pairs = {}

            for j in range(len(tokens) - 1):
                pair       = (tokens[j], tokens[j + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            if pairs[best_pair] < 2:
                break

            new_token  = 256 + len(self.merges)
            new_tokens = []
            j          = 0

            while j < len(tokens):
                if (
                    j + 1 < len(tokens)
                    and (tokens[j], tokens[j + 1]) == best_pair
                ):
                    new_tokens.append(new_token)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1

            tokens = new_tokens
            self.merges.append(best_pair)

            if (i + 1) % 8 == 0:
                print(f"Fusões: {i + 1}/{max_merges} | Tokens: {len(tokens)}")

        self.build_vocab()

    def build_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(self.merges):
            self.vocab[256 + rank] = self.vocab[a] + self.vocab[b]

    def vocab_size(self):
        return 256 + len(self.merges)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            new_token = 256 + rank
            result    = []
            j         = 0
            while j < len(tokens):
                if (
                    j + 1 < len(tokens)
                    and (tokens[j], tokens[j + 1]) == pair
                ):
                    result.append(new_token)
                    j += 2
                else:
                    result.append(tokens[j])
                    j += 1
            tokens = result
        return tokens

    def decode(self, tokens):
        result = b""
        for token in tokens:
            if token in self.vocab:
                result += self.vocab[token]
        return result.decode("utf-8", errors="replace")


# ============================================================
# MODELO GPT
# ============================================================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k       = self.key(x)
        q       = self.query(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()
        head_size   = N_EMBD // N_HEAD
        self.heads  = nn.ModuleList([Head(head_size) for _ in range(N_HEAD)])
        self.projection = nn.Linear(N_EMBD, N_EMBD)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(output))


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.attention    = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.ln1          = nn.LayerNorm(N_EMBD)
        self.ln2          = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class ZoldaxGPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks             = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.final_norm         = nn.LayerNorm(N_EMBD)
        self.language_head      = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T         = idx.shape
        token_emb    = self.token_embedding(idx)
        position_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x            = token_emb + position_emb
        x            = self.blocks(x)
        x            = self.final_norm(x)
        logits       = self.language_head(x)
        loss         = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=TEMPERATURE, top_k=TOP_K):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / max(temperature, 0.01)
            if top_k:
                k              = min(top_k, logits.shape[-1])
                values, indices = torch.topk(logits, k)
                filtered       = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits         = filtered
            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx        = torch.cat((idx, next_token), dim=1)
        return idx


# ============================================================
# DADOS
# ============================================================

def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def carregar_corpus():
    original    = read_text(TEXT_FILE)
    aprendizado = read_text(LEARNING_FILE)
    partes      = []
    if original:
        partes.append(original)
    if aprendizado:
        partes.append("\n\n# EXEMPLOS APROVADOS PELO USUARIO\n\n" + aprendizado)
    corpus = "\n\n".join(partes)
    if len(corpus) < 100:
        raise ValueError("Poucos dados. Verifique texto.txt e aprendizado.txt.")
    return corpus


def preparar_dados(corpus, tokenizer):
    encoded    = tokenizer.encode(corpus)
    data       = torch.tensor(encoded, dtype=torch.long)
    split      = int(len(data) * 0.90)
    train_data = data[:split]
    val_data   = data[split:]
    print("\n========== DADOS ==========")
    print(f"Caracteres:   {len(corpus)}")
    print(f"Tokens:       {len(data)}")
    print(f"Treinamento:  {len(train_data)}")
    print(f"Validação:    {len(val_data)}")
    if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
        raise ValueError("Dados insuficientes para contexto 64. Adicione mais conteúdo ao corpus.")
    return train_data, val_data


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([data[i:i + BLOCK_SIZE]     for i in ix])
    y  = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ============================================================
# AVALIACAO
# ============================================================

@torch.no_grad()
def avaliar(model, train_data, val_data):
    model.eval()
    results = {}
    for name, data in [("treino", train_data), ("validacao", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y       = get_batch(data)
            _, loss    = model(x, y)
            losses[k]  = loss.item()
        results[name] = losses.mean().item()
    model.train()
    return results


# ============================================================
# CHECKPOINT
# ============================================================

def salvar_checkpoint(model, optimizer, tokenizer, iteration, history):
    checkpoint = {
        "version":        "1.2",
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "merges":         tokenizer.merges,
        "vocab_size":     tokenizer.vocab_size(),
        "iteration":      iteration,
        "history":        history
    }
    temp_file = CHECKPOINT_FILE + ".tmp"
    torch.save(checkpoint, temp_file)
    os.replace(temp_file, CHECKPOINT_FILE)


def carregar_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError("Checkpoint não encontrado. Execute primeiro --treino.")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=True)
    if checkpoint.get("version") != "1.2":
        raise ValueError("Checkpoint incompatível. Faça um treinamento inicial da versão 1.2.")
    tokenizer        = BPETokenizer()
    tokenizer.merges = [tuple(pair) for pair in checkpoint["merges"]]
    tokenizer.build_vocab()
    model            = ZoldaxGPT(checkpoint["vocab_size"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer        = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint.get("iteration", 0)
    history   = checkpoint.get("history", [])
    return model, optimizer, tokenizer, iteration, history


# ============================================================
# TREINAMENTO
# ============================================================

def treinar(continuar=False, steps=DEFAULT_STEPS):
    print("\n====================================")
    print("       ZOLDAX.IA 1.2")
    print("       by Saraiva Digital")
    print("====================================")
    print("Dispositivo:", DEVICE)

    if continuar:
        print("\nCarregando modelo anterior...")
        model, optimizer, tokenizer, initial_iteration, history = carregar_checkpoint()
        print("Modelo carregado.")
        print("Iteração anterior:", initial_iteration)
    else:
        if os.path.exists(CHECKPOINT_FILE):
            print("\nATENÇÃO: o treinamento inicial substituirá o checkpoint atual.")
        corpus    = carregar_corpus()
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, MAX_MERGES)
        model     = ZoldaxGPT(tokenizer.vocab_size()).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        initial_iteration = 0
        history           = []

    corpus              = carregar_corpus()
    train_data, val_data = preparar_dados(corpus, tokenizer)

    print("\n========== CONFIGURAÇÃO ==========")
    print("Vocabulário:", tokenizer.vocab_size())
    print("Contexto:   ", BLOCK_SIZE)
    print("Camadas:    ", N_LAYER)
    print("Cabeças:    ", N_HEAD)
    print("Embedding:  ", N_EMBD)
    print("Iterações desta sessão:", steps)
    print("\nIniciando treinamento...\n")

    end_iteration = initial_iteration + steps

    for iteration in range(initial_iteration, end_iteration):
        if iteration % EVAL_INTERVAL == 0 or iteration == initial_iteration:
            losses = avaliar(model, train_data, val_data)
            history.append({
                "iteration": iteration,
                "train_loss": losses["treino"],
                "val_loss":   losses["validacao"]
            })
            print(
                f"Iteração {iteration:6d} | "
                f"Treino: {losses['treino']:.4f} | "
                f"Validação: {losses['validacao']:.4f}"
            )
            salvar_checkpoint(model, optimizer, tokenizer, iteration, history)

        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    salvar_checkpoint(model, optimizer, tokenizer, end_iteration, history)

    print("\n====================================")
    print("TREINAMENTO FINALIZADO")
    print("====================================")
    print("Iteração acumulada:", end_iteration)
    print("Checkpoint salvo:  ", CHECKPOINT_FILE)

    losses = avaliar(model, train_data, val_data)
    print(f"Perda treino:     {losses['treino']:.4f}")
    print(f"Perda validação:  {losses['validacao']:.4f}")


# ============================================================
# HISTORICO DE CONVERSAS
# ============================================================

def salvar_conversa(pergunta, resposta):
    registro = {
        "data":     datetime.now().isoformat(),
        "pergunta": pergunta,
        "resposta": resposta
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def adicionar_aprendizado(pergunta, resposta):
    with open(LEARNING_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\nVocê: {pergunta}\nZoldax: {resposta}\n")
    print("\nExemplo aprovado e salvo em aprendizado.txt")


# ============================================================
# RESPOSTAS FIXAS — carrega do aprendizado.txt
# ============================================================

def carregar_respostas_fixas():
    respostas = {}
    if not os.path.exists(LEARNING_FILE):
        return respostas
    with open(LEARNING_FILE, "r", encoding="utf-8") as f:
        linhas = f.read().strip().split("\n")
    pergunta_atual = None
    for linha in linhas:
        linha = linha.strip()
        if linha.startswith("Você:"):
            pergunta_atual = linha.replace("Você:", "").strip().lower()
        elif linha.startswith("Zoldax:") and pergunta_atual:
            respostas[pergunta_atual] = linha.replace("Zoldax:", "").strip()
            pergunta_atual = None
    return respostas


# ============================================================
# CHAT
# ============================================================

def conversar():
    try:
        model, _, tokenizer, _, _ = carregar_checkpoint()
        model.eval()
    except Exception as error:
        print("\nNão foi possível carregar o modelo:", error)
        print("Execute: python3 zoldax.py --treino")
        return

    # carrega respostas fixas do aprendizado.txt
    RESPOSTAS_FIXAS = carregar_respostas_fixas()

    print("""
╔═══════════════════════════════════════╗
║         <Z O L D A X . I A>           ║
║       Inteligência Artificial         ║
║         by Saraiva Digital.           ║
╚═══════════════════════════════════════╝""")

    print("Comandos:")
    print("/ensinar  - cadastrar um exemplo correto")
    print("/corrigir - corrigir a última resposta")
    print("/sair     - encerrar conversa")
    print("------------------------------------")

    ultima_pergunta = None
    ultima_resposta = None

    while True:
        try:
            pergunta = input("\nVocê: ").strip()

            if not pergunta:
                continue

            if pergunta.lower() in ["/sair", "sair", "exit"]:
                print("\nZoldax: Até logo!")
                break

            if pergunta.lower() == "/ensinar":
                nova_pergunta = input("Pergunta de exemplo: ").strip()
                nova_resposta = input("Resposta correta: ").strip()
                if nova_pergunta and nova_resposta:
                    adicionar_aprendizado(nova_pergunta, nova_resposta)
                    # atualiza dicionário em memória imediatamente
                    RESPOSTAS_FIXAS[nova_pergunta.lower()] = nova_resposta
                continue

            if pergunta.lower() == "/corrigir":
                if not ultima_pergunta:
                    print("Ainda não existe resposta para corrigir.")
                    continue
                resposta_correta = input("Digite a resposta correta: ").strip()
                if resposta_correta:
                    adicionar_aprendizado(ultima_pergunta, resposta_correta)
                    # atualiza dicionário em memória imediatamente
                    RESPOSTAS_FIXAS[ultima_pergunta.lower()] = resposta_correta
                continue

            # ── verifica resposta fixa primeiro ──────────────
            chave = pergunta.lower()
            if chave in RESPOSTAS_FIXAS:
                resposta = RESPOSTAS_FIXAS[chave]
                print("\nZoldax:", resposta)
                salvar_conversa(pergunta, resposta)
                ultima_pergunta = pergunta
                ultima_resposta = resposta
                continue

            # ── geração pelo modelo ───────────────────────────
            prompt  = "\n\nVocê: " + pergunta + "\nZoldax:"
            encoded = tokenizer.encode(prompt)

            input_tensor = torch.tensor(
                [encoded], dtype=torch.long, device=DEVICE
            )

            with torch.no_grad():
                generated = model.generate(input_tensor, MAX_NEW_TOKENS)

            answer_tokens = generated[0].tolist()[len(encoded):]
            resposta      = tokenizer.decode(answer_tokens)
            resposta      = resposta.split("Você:")[0].strip()

            print("\nZoldax:", resposta)
            salvar_conversa(pergunta, resposta)
            ultima_pergunta = pergunta
            ultima_resposta = resposta

        except KeyboardInterrupt:
            print("\nEncerrando Zoldax.")
            break

        except Exception as error:
            print("\nErro:", error)


# ============================================================
# INICIALIZACAO
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zoldax.IA 1.2 - Saraiva Digital")
    parser.add_argument("--treino",    action="store_true", help="Iniciar treinamento do zero")
    parser.add_argument("--continue", dest="continue_training", action="store_true", help="Continuar treinamento anterior")
    parser.add_argument("--steps",    type=int, default=DEFAULT_STEPS, help="Quantidade de iteracoes")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps precisa ser maior que zero")
    if args.treino and args.continue_training:
        parser.error("Escolha --treino ou --continue")

    if args.treino:
        treinar(False, args.steps)
    elif args.continue_training:
        treinar(True, args.steps)
    else:
        conversar()
