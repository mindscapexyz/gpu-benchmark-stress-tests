import os
import time
import random
from locust import HttpUser, task, between, events

# =========================
# Config
# =========================
API_KEY = os.getenv("API_KEY", "token")
MODEL   = os.getenv("MODEL", "microsoft/Phi-3-mini-4k-instruct") # change this to the model you want to test, microsoft/Phi-3-mini-4k-instruct or Qwen/Qwen2.5-7B-Instruct-GPTQ

# Workload knobs
MAX_TOKENS_SHORT = int(os.getenv("MAX_TOKENS_SHORT", "256"))
MAX_TOKENS_RAG   = int(os.getenv("MAX_TOKENS_RAG", "256"))
LONG_PROMPT_TOKS = int(os.getenv("LONG_PROMPT_TOKS", "16000"))  # rough target
RAG_PROMPT_TOKS  = int(os.getenv("RAG_PROMPT_TOKS", "2000"))

# Sample prompts
BASE_SHORT = [
    "Write a concise tip about Git branching.",
    "Explain KV cache like I'm 12.",
    "List 5 ways to speed up LLM inference.",
    "Summarize the benefits of data parallelism for LLM serving.",
    "Explain what speculative decoding is."
]
BASE_RAG = (
    "You are given the following context:\n"
    "{ctx}\n\n"
    "Answer the question clearly and cite relevant facts from the context."
)

def make_repeated_text(words: int) -> str:
    seed = ("NVLink HBM tensor parallelism data parallelism vLLM scheduler "
            "KV cache attention prefill decode tokenization batching throughput latency ")
    text = (seed * ((words * 5) // len(seed) + 1)).strip()
    return text

def approx_tokens_to_words(tokens: int) -> int:
    # crude conversion: ~0.75 tokens per word
    return max(1, int(tokens / 0.75))

def rag_prompt():
    ctx = make_repeated_text(approx_tokens_to_words(RAG_PROMPT_TOKS))
    return BASE_RAG.format(ctx=ctx)


# =========================
# Helper: send request + log TPS
# =========================
def post_chat(client, payload, name):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    t0 = time.time()
    with client.post("/v1/chat/completions",
                     name=name,
                     headers=headers,
                     json=payload,
                     catch_response=True) as resp:
        elapsed = time.time() - t0
        if resp.status_code != 200:
            resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return

        try:
            data = resp.json()
        except Exception as e:
            resp.failure(f"Invalid JSON: {e}")
            return

        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 else 0.0

        # Custom metric row in Locust UI
        events.request.fire(
            request_type="GEN",
            name="tokens_per_second",
            response_time=tps * 1000.0,   # encode TPS as ms (so p50/p95 columns show TPS*1000)
            response_length=completion_tokens,
            context={"tps": tps, "elapsed": elapsed},
            response=None,
            exception=None
        )
        resp.success()

# =========================
# Locust User
# =========================
class LLMUser(HttpUser):
    """
    Run with tags to select workload:
      --tags short   (short-chat)
      --tags rag     (rag-mid)
    """
    wait_time = between(0.05, 0.2)

    @task(2)
    def short_chat(self):
        prompt = random.choice(BASE_SHORT)
        payload = {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS_SHORT,
        }
        post_chat(self.client, payload, name="short_chat")
    short_chat.tags = ["short"]

    @task(1)
    def rag_mid(self):
        prompt = rag_prompt()
        payload = {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Answer strictly from the provided context."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS_RAG,
        }
        post_chat(self.client, payload, name="rag_mid")
    rag_mid.tags = ["rag"]
