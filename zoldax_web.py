import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from flask import Flask, request, jsonify, render_template_string

# ============================================================
# ZOLDAX.IA 1.2 - by Saraiva Digital
# Interface Web
# ============================================================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TEXT_FILE       = os.path.join(BASE_DIR, "texto.txt")
LEARNING_FILE   = os.path.join(BASE_DIR, "aprendizado.txt")
HISTORY_FILE    = os.path.join(BASE_DIR, "conversas.jsonl")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "zoldax_bpe.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE     = 8
BLOCK_SIZE     = 64
N_EMBD         = 64
N_HEAD         = 4
N_LAYER        = 2
DROPOUT        = 0.1
LEARNING_RATE  = 3e-4
MAX_MERGES     = 32
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
                if j + 1 < len(tokens) and (tokens[j], tokens[j + 1]) == pair:
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
# MODELO
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
        w       = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        w       = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w       = F.softmax(w, dim=-1)
        w       = self.dropout(w)
        return w @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size       = N_EMBD // N_HEAD
        self.heads      = nn.ModuleList([Head(head_size) for _ in range(N_HEAD)])
        self.projection = nn.Linear(N_EMBD, N_EMBD)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.projection(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT)
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
        x            = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        x            = self.final_norm(self.blocks(x))
        logits       = self.language_head(x)
        loss         = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=TEMPERATURE, top_k=TOP_K):
        for _ in range(max_new_tokens):
            idx_cond        = idx[:, -BLOCK_SIZE:]
            logits, _       = self(idx_cond)
            logits          = logits[:, -1, :] / max(temperature, 0.01)
            if top_k:
                k               = min(top_k, logits.shape[-1])
                values, indices = torch.topk(logits, k)
                filtered        = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits          = filtered
            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx        = torch.cat((idx, next_token), dim=1)
        return idx

# ============================================================
# CARREGAMENTO
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

def carregar_modelo():
    if not os.path.exists(CHECKPOINT_FILE):
        return None, None, {}
    ckpt      = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=True)
    tokenizer = BPETokenizer()
    tokenizer.merges = [tuple(p) for p in ckpt["merges"]]
    tokenizer.build_vocab()
    model     = ZoldaxGPT(ckpt["vocab_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    respostas = carregar_respostas_fixas()
    return model, tokenizer, respostas

def salvar_conversa(pergunta, resposta):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "data": datetime.now().isoformat(),
            "pergunta": pergunta,
            "resposta": resposta
        }, ensure_ascii=False) + "\n")

def adicionar_aprendizado(pergunta, resposta):
    with open(LEARNING_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\nVocê: {pergunta}\nZoldax: {resposta}\n")

# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

model, tokenizer, RESPOSTAS_FIXAS = carregar_modelo()

HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zoldax.IA</title>
<style>
  :root {
    --bg:        #1a1a2e;
    --sidebar:   #16213e;
    --surface:   #0f3460;
    --accent:    #e94560;
    --accent2:   #533483;
    --text:      #eaeaea;
    --text-dim:  #8892a4;
    --user-bg:   #1e3a5f;
    --bot-bg:    #1a1a2e;
    --input-bg:  #16213e;
    --border:    #2a2a4a;
    --radius:    12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; overflow: hidden; }

  /* SIDEBAR */
  .sidebar { width: 260px; background: var(--sidebar); display: flex; flex-direction: column; border-right: 1px solid var(--border); flex-shrink: 0; }
  .sidebar-logo { padding: 24px 20px 16px; border-bottom: 1px solid var(--border); }
  .sidebar-logo h1 { font-size: 1.3rem; font-weight: 700; letter-spacing: 2px; }
  .sidebar-logo h1 span { color: var(--accent); }
  .sidebar-logo p { font-size: 0.72rem; color: var(--text-dim); margin-top: 4px; letter-spacing: 1px; }
  .sidebar-btn { margin: 16px 12px 0; background: var(--accent); border: none; color: #fff; padding: 10px 16px; border-radius: var(--radius); cursor: pointer; font-size: 0.85rem; font-weight: 600; letter-spacing: 1px; transition: opacity .2s; }
  .sidebar-btn:hover { opacity: .85; }
  .sidebar-section { padding: 20px 16px 8px; font-size: 0.7rem; color: var(--text-dim); letter-spacing: 1px; text-transform: uppercase; }
  .sidebar-info { padding: 0 16px; font-size: 0.78rem; color: var(--text-dim); line-height: 1.6; }
  .sidebar-info span { color: var(--text); font-weight: 500; }
  .sidebar-footer { margin-top: auto; padding: 16px; border-top: 1px solid var(--border); font-size: 0.72rem; color: var(--text-dim); text-align: center; }

  /* MAIN */
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .topbar { padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; background: var(--sidebar); }
  .topbar-dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 6px #22c55e; }
  .topbar span { font-size: 0.85rem; color: var(--text-dim); }

  /* CHAT */
  .chat-area { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; }
  .chat-area::-webkit-scrollbar { width: 6px; }
  .chat-area::-webkit-scrollbar-track { background: transparent; }
  .chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .msg { display: flex; gap: 12px; max-width: 780px; animation: fadeIn .3s ease; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.bot  { align-self: flex-start; }

  .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; flex-shrink: 0; }
  .avatar.user-av { background: var(--accent2); color: #fff; }
  .avatar.bot-av  { background: var(--accent);  color: #fff; letter-spacing: 1px; }

  .bubble { padding: 12px 16px; border-radius: var(--radius); line-height: 1.65; font-size: 0.92rem; max-width: 640px; }
  .msg.user .bubble { background: var(--user-bg); border-bottom-right-radius: 4px; }
  .msg.bot  .bubble { background: var(--bot-bg); border: 1px solid var(--border); border-bottom-left-radius: 4px; }

  .typing { display: flex; gap: 5px; align-items: center; padding: 14px 16px; }
  .typing span { width: 7px; height: 7px; background: var(--text-dim); border-radius: 50%; animation: bounce 1.2s infinite; }
  .typing span:nth-child(2) { animation-delay: .2s; }
  .typing span:nth-child(3) { animation-delay: .4s; }
  @keyframes bounce { 0%,80%,100%{transform:scale(0.7);opacity:.5} 40%{transform:scale(1);opacity:1} }

  /* INPUT */
  .input-area { padding: 16px 24px 20px; border-top: 1px solid var(--border); background: var(--sidebar); }
  .input-wrap { display: flex; gap: 10px; align-items: flex-end; background: var(--input-bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 14px; transition: border-color .2s; }
  .input-wrap:focus-within { border-color: var(--accent); }
  textarea { flex:1; background: transparent; border: none; outline: none; color: var(--text); font-size: 0.92rem; resize: none; min-height: 24px; max-height: 140px; font-family: inherit; line-height: 1.5; }
  textarea::placeholder { color: var(--text-dim); }
  .send-btn { background: var(--accent); border: none; color: #fff; width: 36px; height: 36px; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: opacity .2s; }
  .send-btn:hover { opacity: .85; }
  .send-btn:disabled { opacity: .4; cursor: default; }
  .send-btn svg { width: 16px; height: 16px; }
  .input-hint { text-align: center; font-size: 0.7rem; color: var(--text-dim); margin-top: 8px; }

  /* ENSINAR MODAL */
  .modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.6); z-index:100; align-items:center; justify-content:center; }
  .modal-overlay.open { display:flex; }
  .modal { background: var(--sidebar); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; width: 420px; display: flex; flex-direction: column; gap: 14px; }
  .modal h2 { font-size: 1rem; font-weight: 600; }
  .modal label { font-size: 0.8rem; color: var(--text-dim); margin-bottom: 4px; display: block; }
  .modal input, .modal textarea { width:100%; background: var(--input-bg); border: 1px solid var(--border); color: var(--text); border-radius: 8px; padding: 10px 12px; font-size: 0.88rem; font-family: inherit; outline:none; }
  .modal input:focus, .modal textarea:focus { border-color: var(--accent); }
  .modal-actions { display:flex; gap:10px; justify-content:flex-end; margin-top:4px; }
  .btn-cancel { background: transparent; border: 1px solid var(--border); color: var(--text-dim); padding: 8px 18px; border-radius: 8px; cursor:pointer; font-size:0.85rem; }
  .btn-save   { background: var(--accent); border:none; color:#fff; padding: 8px 18px; border-radius: 8px; cursor:pointer; font-size:0.85rem; font-weight:600; }

  /* WELCOME */
  .welcome { margin: auto; text-align: center; color: var(--text-dim); padding: 40px 20px; }
  .welcome h2 { font-size: 1.6rem; color: var(--text); margin-bottom: 10px; letter-spacing: 2px; }
  .welcome p  { font-size: 0.9rem; line-height: 1.7; max-width: 400px; }
</style>
</head>
<body>

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <h1>&lt;<span>ZOLDAX</span>.IA&gt;</h1>
    <p>by Saraiva Digital</p>
  </div>
  <button class="sidebar-btn" onclick="novaConversa()">+ Nova conversa</button>
  <div class="sidebar-section">Sistema</div>
  <div class="sidebar-info">
    Dispositivo: <span id="device-info">carregando...</span><br>
    Modelo: <span>ZoldaxGPT v1.2</span><br>
    Status: <span id="status-info">Online</span>
  </div>
  <div class="sidebar-section">Ações</div>
  <div style="padding: 0 12px; display:flex; flex-direction:column; gap:8px;">
    <button class="sidebar-btn" style="background:var(--accent2); margin:0;" onclick="abrirModal()">✏️ Ensinar resposta</button>
  </div>
  <div class="sidebar-footer">Zoldax.IA 1.2 [<0>]</div>
</aside>

<!-- MAIN -->
<main class="main">
  <div class="topbar">
    <div class="topbar-dot"></div>
    <span>Zoldax.IA está Online</span>
  </div>

  <div class="chat-area" id="chat"></div>

  <div class="input-area">
    <div class="input-wrap">
      <textarea id="inp" placeholder="Digite sua mensagem..." rows="1" onkeydown="tecla(event)" oninput="ajustar(this)"></textarea>
      <button class="send-btn" id="sendBtn" onclick="enviar()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
      </button>
    </div>
    <div class="input-hint">Enter para enviar · Shift+Enter para nova linha</div>
  </div>
</main>

<!-- MODAL ENSINAR -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <h2>✏️ Ensinar resposta</h2>
    <div>
      <label>Pergunta</label>
      <input type="text" id="modal-perg" placeholder="quem é você?">
    </div>
    <div>
      <label>Resposta correta</label>
      <textarea id="modal-resp" rows="3" placeholder="Sou Zoldax.IA, by Saraiva Digital."></textarea>
    </div>
    <div class="modal-actions">
      <button class="btn-cancel" onclick="fecharModal()">Cancelar</button>
      <button class="btn-save" onclick="salvarEnsino()">Salvar</button>
    </div>
  </div>
</div>

<script>
const chat    = document.getElementById("chat");
const inp     = document.getElementById("inp");
const sendBtn = document.getElementById("sendBtn");

let ultimaResposta = null;
let ultimaPergunta = null;
let primeiraMsg    = true;

// info do dispositivo
fetch("/api/info").then(r=>r.json()).then(d=>{
  document.getElementById("device-info").textContent = d.device;
});

function ajustar(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}

function tecla(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); enviar(); }
}

function novaConversa() {
  chat.innerHTML = "";
  primeiraMsg    = true;
  ultimaPergunta = null;
  ultimaResposta = null;
}

function addMsg(texto, tipo) {
  if (primeiraMsg) {
    const w = document.querySelector(".welcome");
    if (w) w.remove();
    primeiraMsg = false;
  }
  const msg = document.createElement("div");
  msg.className = "msg " + tipo;
  const av  = document.createElement("div");
  av.className = "avatar " + (tipo === "user" ? "user-av" : "bot-av");
  av.textContent = tipo === "user" ? "EU" : "ZX";
  const bub = document.createElement("div");
  bub.className = "bubble";
  bub.textContent = texto;
  msg.appendChild(av);
  msg.appendChild(bub);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return bub;
}

function addTyping() {
  const msg = document.createElement("div");
  msg.className = "msg bot";
  msg.id = "typing";
  msg.innerHTML = `<div class="avatar bot-av">ZX</div><div class="bubble typing"><span></span><span></span><span></span></div>`;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById("typing");
  if (t) t.remove();
}

async function enviar() {
  const texto = inp.value.trim();
  if (!texto) return;
  inp.value = "";
  inp.style.height = "auto";
  sendBtn.disabled = true;

  addMsg(texto, "user");
  ultimaPergunta = texto;
  addTyping();

  try {
    const res  = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pergunta: texto })
    });
    const data = await res.json();
    removeTyping();
    addMsg(data.resposta, "bot");
    ultimaResposta = data.resposta;
  } catch(e) {
    removeTyping();
    addMsg("Erro ao conectar com Zoldax.", "bot");
  }

  sendBtn.disabled = false;
  inp.focus();
}

function abrirModal() {
  if (ultimaPergunta) document.getElementById("modal-perg").value = ultimaPergunta;
  document.getElementById("modal").classList.add("open");
}

function fecharModal() {
  document.getElementById("modal").classList.remove("open");
}

async function salvarEnsino() {
  const perg = document.getElementById("modal-perg").value.trim();
  const resp = document.getElementById("modal-resp").value.trim();
  if (!perg || !resp) return;
  await fetch("/api/ensinar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pergunta: perg, resposta: resp })
  });
  fecharModal();
  document.getElementById("modal-perg").value = "";
  document.getElementById("modal-resp").value = "";
  addMsg(`✅ Aprendi: "${perg}" → "${resp}"`, "bot");
}

// welcome inicial
(function() {
  const w = document.createElement("div");
  w.className = "welcome";
  w.innerHTML = `<h2>&lt;ZOLDAX.IA&gt;</h2><p>Inteligência artificial by Saraiva Digital.<br>[<0>]</p>`;
  chat.appendChild(w);
})();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/info")
def info():
    return jsonify({"device": DEVICE})

@app.route("/api/chat", methods=["POST"])
def chat():
    if model is None:
        return jsonify({"resposta": "Modelo não carregado. Execute --train primeiro."})

    data     = request.get_json()
    pergunta = data.get("pergunta", "").strip()
    if not pergunta:
        return jsonify({"resposta": ""})

    # resposta fixa tem prioridade
    chave = pergunta.lower()
    if chave in RESPOSTAS_FIXAS:
        resposta = RESPOSTAS_FIXAS[chave]
        salvar_conversa(pergunta, resposta)
        return jsonify({"resposta": resposta})

    # geração pelo modelo
    prompt   = "\n\nVocê: " + pergunta + "\nZoldax:"
    encoded  = tokenizer.encode(prompt)
    tensor   = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        generated = model.generate(tensor, MAX_NEW_TOKENS)

    tokens   = generated[0].tolist()[len(encoded):]
    resposta = tokenizer.decode(tokens)
    resposta = resposta.split("Você:")[0].strip()

    salvar_conversa(pergunta, resposta)
    return jsonify({"resposta": resposta})

@app.route("/api/ensinar", methods=["POST"])
def ensinar():
    data     = request.get_json()
    pergunta = data.get("pergunta", "").strip()
    resposta = data.get("resposta", "").strip()
    if pergunta and resposta:
        adicionar_aprendizado(pergunta, resposta)
        RESPOSTAS_FIXAS[pergunta.lower()] = resposta
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("\n====================================")
    print("       ZOLDAX.IA 1.2 - WEB")
    print("       by Saraiva Digital")
    print("====================================")
    if model is None:
        print("AVISO: modelo não encontrado. Execute --train no zoldax.py primeiro.")
    else:
        print("Modelo carregado com sucesso.")
    print(f"Dispositivo: {DEVICE}")
    print("\nAcesse em qualquer PC da rede:")
    print("  http://<IP-DESTE-PC>:5000")
    print("\nCtrl+C para encerrar.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
