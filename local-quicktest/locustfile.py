import os
import time
import random
from locust import HttpUser, task, between, events

# =========================
# Config
# =========================
API_KEY = os.getenv("API_KEY", "token")
MODEL   = os.getenv("MODEL", "microsoft/Phi-3-mini-4k-instruct") # microsoft/Phi-3-mini-4k-instruct or Qwen/Qwen2.5-7B-Instruct-GPTQ

# Workload knobs
MAX_TOKENS_SHORT = int(os.getenv("MAX_TOKENS_SHORT", "256"))
MAX_TOKENS_RAG   = int(os.getenv("MAX_TOKENS_RAG", "256"))
RAG_PROMPT_TOKS  = int(os.getenv("RAG_PROMPT_TOKS", "2000"))

# Prompts
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
    return (seed * ((words * 5) // len(seed) + 1)).strip()

def approx_tokens_to_words(tokens: int) -> int:
    return max(1, int(tokens / 0.75))

def rag_prompt():
    ctx = make_repeated_text(approx_tokens_to_words(RAG_PROMPT_TOKS))
    return BASE_RAG.format(ctx=ctx)

# =========================
# Helpers
# =========================
def post_chat_non_streaming(client, payload, name):
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

        data = resp.json()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 else 0.0

        # Custom metric row for TPS
        events.request.fire(
            request_type="GEN",
            name="tokens_per_second",
            response_time=tps * 1000.0,  # encode TPS as ms
            response_length=completion_tokens,
            context={"tps": tps, "elapsed": elapsed},
            response=None,
            exception=None
        )
        resp.success()

def post_chat_streaming_ttft(client, payload, name):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = dict(payload)
    payload["stream"] = True

    t0 = time.time()
    with client.post("/v1/chat/completions",
                     name=name,
                     headers=headers,
                     json=payload,
                     stream=True,
                     catch_response=True) as resp:
        if resp.status_code != 200:
            resp.failure(f"HTTP {resp.status_code}: {getattr(resp, 'text', '')[:200]}")
            return

        ttft = None
        try:
            for raw in resp.iter_lines(decode_unicode=True, chunk_size=1):
                if not raw or not str(raw).startswith("data:"):
                    continue
                if "[DONE]" in raw:
                    break
                if ttft is None:
                    ttft = (time.time() - t0) * 1000.0  # ms
                    events.request.fire(
                        request_type="GEN",
                        name="ttft_ms",
                        response_time=ttft,
                        response_length=0,
                        context={"ttft_ms": ttft},
                        response=None,
                        exception=None
                    )
        except Exception as e:
            resp.failure(f"Stream error: {e}")
            return

        resp.success()

# =========================
# User
# =========================
class LLMUser(HttpUser):
    wait_time = between(0.05, 0.2)

    # ---- Non-streaming throughput tasks ----
    @task(3)
    def short_chat(self):
        payload = {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": random.choice(BASE_SHORT)}
            ],
            "max_tokens": MAX_TOKENS_SHORT,
        }
        post_chat_non_streaming(self.client, payload, name="short_chat")

    @task(2)
    def rag_mid(self):
        payload = {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Answer strictly from the provided context."},
                {"role": "user", "content": rag_prompt()}
            ],
            "max_tokens": MAX_TOKENS_RAG,
        }
        post_chat_non_streaming(self.client, payload, name="rag_mid")

    # ---- Streaming TTFT tasks ----
    @task(1)
    def ttft_short(self):
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Respond briefly."},
                {"role": "user", "content": "Say hello in one short sentence."}
            ],
            "max_tokens": 64,
        }
        post_chat_streaming_ttft(self.client, payload, name="ttft_short")

    @task(1)
    def ttft_rag(self):
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Answer from context."},
                {"role": "user", "content": rag_prompt()}
            ],
            "max_tokens": 128,
        }
        post_chat_streaming_ttft(self.client, payload, name="ttft_rag")
