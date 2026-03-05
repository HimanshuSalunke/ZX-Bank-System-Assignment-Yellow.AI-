# ZX Bank — Conversational AI Backend + Chat UI

> A production-grade conversational AI system for ZX Bank that answers customer queries using **Hybrid RAG** (BM25 + FAISS), **multi-turn conversation memory**, **adversarial safety**, and **human escalation** — powered by **GPT-4.1-nano** via **Requesty**, with **GPU-accelerated** local processing on NVIDIA RTX 4050.

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Python Version Requirement](#-python-version-requirement)
- [GPU Acceleration](#-gpu-acceleration)
- [AI/ML Models & Methods](#-aiml-models--methods)
- [LLM Provider — Why Requesty?](#-llm-provider--why-requesty)
- [Model Selection — Why GPT-4.1-nano?](#-model-selection--why-gpt-41-nano)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
- [Retrieval Strategy](#-retrieval-strategy)
- [Query Classification & Safety](#-query-classification--safety)
- [Human Escalation Workflow](#-human-escalation-workflow)
- [Sample Queries](#-sample-queries)
- [Observability & Logging](#-observability--logging)

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        User / Chat UI                            │
│                     (SSE Streaming, Port 8000)                   │
└───────────────┬──────────────────────────────────────────────────┘
                │ POST /api/chat/stream (Server-Sent Events)
                ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI Server                               │
│                                                                  │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │   Session   │   │    Query     │   │   Adversarial        │   │
│  │  Manager    │──▶│  Classifier  │──▶│   Safety Filter      │   │
│  └────────────┘   └──────┬───────┘   └──────────────────────┘   │
│                          │                                       │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                      │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────┐            │
│  │ Small Talk   │ │  Document    │ │  Escalation   │            │
│  │ Handler      │ │  Handler     │ │  Handler      │            │
│  │ (LLM stream) │ │ (RAG+stream) │ │ (Stateful)    │            │
│  └─────────────┘ └──────┬───────┘ └───────────────┘            │
│                          │                                       │
│              ┌──────────────────────┐                           │
│              │   Hybrid Retriever   │                           │
│              │  BM25 + FAISS + RRF  │                           │
│              └──────────┬───────────┘                           │
│                         │                                        │
│         ┌───────────────┼────────────────┐                      │
│         ▼               ▼                ▼                       │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐             │
│  │ BM25 Index │  │ FAISS Index │  │ SBERT GPU    │             │
│  │  (Sparse)  │  │  (Dense)    │  │ (Embeddings) │             │
│  └────────────┘  └─────────────┘  └──────────────┘             │
│                                                                  │
│              ┌──────────────────────┐                           │
│              │   Requesty API       │                           │
│              │ (GPT-4.1-nano)       │                           │
│              │ SSE Token Streaming  │                           │
│              └──────────────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Runtime** | Python 3.10.0 | Core runtime (see [version note](#-python-version-requirement)) |
| **API** | FastAPI + Uvicorn | Async HTTP server with SSE streaming |
| **LLM** | GPT-4.1-nano (via Requesty) | Response generation (streamed) |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Dense vector embeddings (**GPU**) |
| **GPU** | NVIDIA RTX 4050 (6GB VRAM, CUDA) | GPU-accelerated embeddings |
| **Vector Search** | FAISS (IndexFlatIP) | Dense semantic retrieval |
| **Sparse Search** | BM25 (Okapi) | Keyword-based retrieval |
| **Keyword Extraction** | TF-IDF (scikit-learn) | Automated keyword extraction |
| **Re-Ranking** | Reciprocal Rank Fusion (RRF) | Hybrid result fusion |
| **Doc Splitting** | LangChain MarkdownHeaderTextSplitter | Structure-preserving chunking |
| **Config** | Pydantic Settings + .env | Type-safe configuration |
| **Logging** | structlog + rich | Structured observability |
| **Frontend** | HTML + CSS + JS | Premium dark-mode chat UI with SSE |

---

## 🐍 Python Version Requirement

> **This project requires Python 3.10.0**

**Reason**: The development machine runs **NVIDIA RTX 4050** with **GPU-accelerated TensorFlow** for deep learning workloads. TensorFlow's GPU support on Windows requires **Python 3.10** specifically — newer Python versions (3.11+) are not compatible with TensorFlow GPU on Windows as of the current release cycle. Since this project uses GPU-accelerated processing (via CUDA) for the Sentence-Transformer embedding model, Python 3.10.0 ensures full compatibility across the entire GPU compute stack:

- **CUDA Toolkit** → **cuDNN** → **TensorFlow GPU** → **PyTorch GPU** → **sentence-transformers**

All project dependencies are pinned to their latest versions that support Python 3.10.

---

## ⚡ GPU Acceleration

This project leverages the **NVIDIA GeForce RTX 4050 (6GB VRAM)** for accelerated processing:

| Component | Runs On | Why |
|---|---|---|
| **SBERT Embeddings** (all-MiniLM-L6-v2) | **GPU** (CUDA) | Embedding generation is ~10x faster on GPU vs CPU. Critical for both index building (182 chunks) and real-time query embedding. |
| **TF-IDF** (scikit-learn) | CPU | TF-IDF is a sparse matrix operation — inherently CPU-bound. With 182 chunks, it completes in <50ms. GPU overhead would be counterproductive. |
| **BM25** (rank-bm25) | CPU | BM25 is a simple scoring function over inverted indices. Sub-millisecond on CPU. No GPU implementation exists. |
| **FAISS** (IndexFlatIP) | CPU | With only 182 vectors (384-dim), CPU FAISS takes ~1ms. `faiss-gpu` adds 100ms+ CUDA kernel launch overhead — net slower for small indexes. GPU FAISS becomes beneficial at 100K+ vectors. |
| **RRF Re-Ranking** | CPU | Pure arithmetic over ~10 results. Microseconds on CPU. |
| **LLM** (GPT-4.1-nano) | **Remote API** | Runs on OpenAI's infrastructure via Requesty router. Streaming delivers first token in ~0.5-1 second. |

> **Engineering Note**: GPU is used where it provides measurable speedup (embeddings). For components where the dataset is small (182 chunks) or the algorithm is inherently sparse/sequential (BM25, TF-IDF), CPU execution is faster than GPU due to avoided CUDA kernel launch and memory transfer overhead. This is the production-correct approach.

---

## 🤖 AI/ML Models & Methods

### Local Models (run on your machine)

| Model / Method | Type | Parameters | What It Does | Runs On |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | Sentence-BERT (Transformer) | 22.7M | Converts text → 384-dimensional dense vectors for semantic similarity search | **RTX 4050 GPU** |
| **TF-IDF** (TfidfVectorizer) | Statistical NLP | N/A | Extracts top-10 keywords per document chunk using bigram TF-IDF scoring. Used for metadata enrichment. | CPU |
| **BM25** (Okapi BM25) | Probabilistic IR | N/A | Ranks documents by keyword relevance using term frequency, inverse document frequency, and document length normalization. | CPU |
| **FAISS** (IndexFlatIP) | Vector Search | N/A | Brute-force cosine similarity search over normalized embeddings. Returns top-K nearest neighbors. | CPU |
| **Reciprocal Rank Fusion** | Ensemble Method | N/A | Combines BM25 (sparse) and FAISS (dense) rankings into a unified score using `RRF(d) = Σ 1/(k + rank(d))` where k=60. | CPU |

### Remote Model (API call)

| Model | Provider | What It Does | Latency |
|---|---|---|---|
| **GPT-4.1-nano** | OpenAI via Requesty | Generates natural language responses from retrieved context. Streamed token-by-token via SSE. | ~0.5-1s first token |

> **Important**: No model training happens anywhere in this system. All models are **pre-trained**. We only perform **inference** (forward pass) on the local SBERT model and make API calls to the remote LLM.

---

## 🔗 LLM Provider — Why Requesty?

> **Note**: The assignment suggested using OpenRouter as the LLM provider. However, I encountered **payment/card processing issues** with OpenRouter's platform that prevented me from using it.

I chose **[Requesty](https://requesty.ai)** (`router.requesty.ai`) as the LLM router instead. Requesty is a more reliable alternative that offers:

- ✅ **Near-zero downtime** for all models with automatic failover
- ✅ **Great latency** with intelligent routing and latency-based model selection
- ✅ **300+ models** including all major providers (OpenAI, Anthropic, Google, Mistral)
- ✅ **OpenAI SDK compatible** — uses the standard `openai` Python package
- ✅ **SSE streaming support** for real-time token delivery
- ✅ **Cost tracking** and spend management

### Provider-Agnostic Design

The system is **fully provider-agnostic**. Switching to OpenRouter or any other OpenAI-compatible endpoint requires only changing two environment variables:

```env
LLM_BASE_URL=https://openrouter.ai/api/v1   # or any other provider
REQUESTY_API_KEY=your_api_key_here
```

No code changes needed.

---

## 🏎 Model Selection — Why GPT-4.1-nano?

For our banking assistant use case, we needed the **fastest and most cost-effective** model that delivers high-quality grounded responses. After evaluating the available models on Requesty:

| Model | First Token Latency | Cost (Input / Output per 1M) | Quality | Verdict |
|---|---|---|---|---|
| `google/gemini-3-flash-preview` | 30-120+ seconds | $0.15 / $0.60 | Good | ❌ Preview model — cold start issues, unreliable latency |
| `openai/gpt-4.1` | 3-5 seconds | $2.00 / $8.00 | Excellent | ❌ Overkill for grounded Q&A, too expensive |
| `openai/gpt-4.1-mini` | ~1-2 seconds | $0.40 / $1.60 | Very Good | ⚠️ Fast but costlier than needed |
| **`openai/gpt-4.1-nano`** | **~0.5-1 second** | **$0.10 / $0.40** | **Good** | **✅ Fastest & cheapest — ideal for grounded RAG** |

**GPT-4.1-nano** was selected because:
- **Speed**: ~0.5-1 second first-token latency — responses start appearing instantly
- **Cost**: 20x cheaper than GPT-4.1, 4x cheaper than mini
- **Quality**: Excellent for our use case because the response quality is driven by the **retrieval pipeline**, not the LLM's raw reasoning. The LLM only needs to synthesize well-retrieved context into natural language — nano handles this perfectly.
- **Reliability**: Production-stable model with consistent sub-second TTFT

Since our system uses **RAG (Retrieval-Augmented Generation)**, the heavy lifting is done by the retrieval pipeline. The LLM's job is to synthesize retrieved context into a natural response — nano handles this perfectly.

---

## 📁 Project Structure

```
├── app/                          # Application source code
│   ├── main.py                   # FastAPI app with lifespan hooks
│   ├── config.py                 # Pydantic settings (secrets from .env)
│   ├── api/
│   │   ├── routes.py             # HTTP endpoints + SSE streaming
│   │   └── schemas.py            # Request/response Pydantic models
│   ├── core/
│   │   ├── classifier.py         # Query type classification
│   │   ├── conversation.py       # Multi-turn memory (sliding window)
│   │   ├── llm.py                # LLM client (sync + streaming)
│   │   └── safety.py             # Adversarial detection
│   ├── retrieval/
│   │   ├── document_processor.py # Markdown loading + splitting
│   │   ├── embeddings.py         # SBERT GPU embeddings
│   │   ├── bm25_index.py         # BM25 sparse index
│   │   ├── tfidf_extractor.py    # TF-IDF keyword extraction
│   │   ├── vector_store.py       # FAISS dense index
│   │   └── hybrid_retriever.py   # BM25 + FAISS + RRF fusion
│   ├── handlers/
│   │   ├── document_handler.py   # RAG-grounded answers
│   │   ├── smalltalk_handler.py  # Casual conversation
│   │   └── escalation_handler.py # Human handoff (stateful)
│   └── utils/
│       ├── logger.py             # structlog + rich setup
│       └── helpers.py            # Shared utilities
├── data/
│   ├── documents/                # 20 ZX Bank knowledge base docs
│   └── escalations/              # Stored escalation records (JSON)
├── indexes/                      # Persisted FAISS + BM25 indexes
├── frontend/                     # Chat UI (HTML + CSS + JS, SSE)
├── queries/                      # Test queries (standard + adversarial)
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
├── run.py                        # One-command startup script
├── setup_index.py                # Index building script
└── README.md                     # This file
```

---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.10.0** (required — see [version note](#-python-version-requirement))
- NVIDIA GPU with CUDA support (optional, falls back to CPU)
- Requesty API key (get one at [requesty.ai](https://requesty.ai))

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env and set your Requesty API key
```

Only **3 settings** in `.env` — everything else is hardcoded as engineering defaults in `config.py`:

```env
REQUESTY_API_KEY=your_key_here          # Required
LLM_MODEL=openai/gpt-4.1-nano          # Default: fastest for banking Q&A
LLM_BASE_URL=https://router.requesty.ai/v1  # Default: Requesty router
```

> **⚠️ For Evaluators**: The API key is shared separately for security. Paste it into the `REQUESTY_API_KEY` field in `.env`. The `.env` file is excluded from Git to prevent accidental secret exposure — this is a security best practice.

### Step 3: Build Indexes

```bash
python setup_index.py
```

This processes all 20 markdown documents, extracts TF-IDF keywords, generates GPU-accelerated SBERT embeddings, and builds both FAISS and BM25 indexes.

```
✅ Indexes built successfully
   → 182 chunks indexed
   → FAISS vectors: 182 (384-dim, GPU-generated)
   → BM25 documents: 182
```

---

## ▶ Running the Application

**Single command to start everything:**

```bash
python run.py
```

This command automatically:
1. ✅ Kills any existing process on port 8000
2. ✅ Checks for indexes (builds them if missing)
3. ✅ Sets HuggingFace offline mode (no model re-downloads)
4. ✅ Starts the FastAPI server with uvicorn

Open **http://localhost:8000** for the chat UI with **real-time SSE streaming**.

> **Note**: The SBERT embedding model (`all-MiniLM-L6-v2`, 80MB) is downloaded once on first run and cached locally in `~/.cache/huggingface/`. All subsequent startups load from this local cache with zero network access.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat/stream` | **Primary** — SSE streaming (word-by-word) |
| `POST` | `/api/chat` | Full response (non-streaming, for compatibility) |
| `GET` | `/api/health` | System health check |
| `GET` | `/api/history/{session_id}` | Conversation history |
| `GET` | `/api/escalations` | List escalation records |

### POST /api/chat/stream (SSE)

Streams tokens in real-time via Server-Sent Events:

```
event: meta
data: {"session_id":"abc123","query_type":"document_query","confidence":0.95,"sources":[...]}

event: token
data: "ZX"

event: token
data: " Bank"

event: token
data: " offers"

event: done
data: {}
```

---

## 🔍 Retrieval Strategy

### Hybrid RAG Pipeline

1. **BM25 (Sparse)**: Keyword-based matching using Okapi BM25. Excels at exact terms (e.g., "CIBIL score 700", "Section 80C").
2. **FAISS (Dense)**: Semantic search using GPU-generated SBERT embeddings. Excels at meaning-based matching (e.g., "how to save tax" → finds tax-saver FD info).
3. **Reciprocal Rank Fusion (RRF)**: `RRF(d) = Σ 1/(k + rank(d))` where `k=60`. Produces a unified ranking leveraging both sparse and dense strengths.

### Document Processing
- **Splitter**: `MarkdownHeaderTextSplitter` preserves heading hierarchy
- **Metadata**: Each chunk carries `doc_title`, `section_heading`, `doc_type`, and TF-IDF `keywords`
- **Embeddings**: 384-dim L2-normalised vectors from all-MiniLM-L6-v2 (GPU)

---

## 🛡 Query Classification & Safety

| Type | Trigger | Handler | Retrieval? |
|---|---|---|---|
| `document_query` | Banking questions | Hybrid RAG + LLM streaming | ✅ Yes |
| `small_talk` | Greetings, thanks | LLM streaming (no retrieval) | ❌ No |
| `escalation` | "speak to agent" | Stateful collection flow | ❌ No |
| `adversarial` | Jailbreak, injection | Instant safe refusal | ❌ No |

### Multi-Layer Adversarial Safety
1. **Pattern Matching**: Regex detection for prompt injection, jailbreak, social engineering
2. **Deep Safety Check**: Secondary analysis on all incoming document queries
3. **Contextual Refusals**: Different refusal messages per attack type

---

## 🤝 Human Escalation Workflow

```
User: "I want to speak to a human"
Bot:  "May I have your full name?"          ← Step 1
User: "Rahul Sharma"
Bot:  "Could you share your phone number?"  ← Step 2
User: "+91-9876543210"
Bot:  "What's the reason for your request?" ← Step 3
User: "Loan stuck for 2 weeks"
Bot:  "✅ Recorded! Reference: C8D3..."      ← Saved to JSON
```

- Validated phone number format (7-15 digits)
- Persisted to `data/escalations/` as JSON
- Retrievable via `GET /api/escalations`

---

## 💬 Sample Queries

### Standard Queries (12)

| # | Query | Expected Topic |
|---|---|---|
| Q01 | What are the interest rates for home loans? | Home Loans |
| Q02 | How do I open a savings account online? | Savings Accounts |
| Q03 | What are the charges for using other bank ATMs? | Fees & Charges |
| Q04 | Tell me about your credit card rewards | Credit Cards |
| Q05 | FD interest rate for senior citizens? | Fixed Deposits |
| Q06 | I lost my debit card. What should I do? | Debit Cards |
| Q07 | How can I register for mobile banking? | Digital Banking |
| Q08 | What documents are needed for education loan? | Education Loans |
| Q09 | How do I report a fraudulent transaction? | Security |
| Q10 | What NRI accounts does the bank offer? | Forex / NRI |
| Q11 | Explain the Sukanya Samriddhi Yojana | Government Schemes |
| Q12 | I want to update my KYC | Account Management |

### Adversarial Queries (3)

| # | Attack Type | Expected Behaviour |
|---|---|---|
| A01 | Prompt Injection | Safe refusal |
| A02 | Jailbreak (DAN) | Safe refusal + redirect |
| A03 | Social Engineering (fake RBI auditor) | Contextual refusal |

---

## 📊 Observability & Logging

Every request produces structured terminal logs via `structlog`:

```
2026-03-04 14:30:00 [info] query_classified   query_type=document_query session_id=a1b2c3
2026-03-04 14:30:00 [info] retrieval_timing   elapsed_seconds=0.005 results=5
2026-03-04 14:30:02 [info] llm_stream_complete first_token_seconds=1.23 tokens_streamed=145 total_seconds=3.8
```

| Log Field | Description |
|---|---|
| `query_type` | Classification result |
| `retrieval_triggered` | Whether RAG pipeline was used |
| `elapsed_seconds` | Retrieval latency |
| `first_token_seconds` | Time to first LLM token |
| `tokens_streamed` | Number of tokens generated |
| `session_id` | Conversation session ID |

---

## 📄 License

This project was built as part of the **ML Intern Assignment** for Yellow.ai.
