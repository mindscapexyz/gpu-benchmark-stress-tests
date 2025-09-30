# if don't have uv installed yet
# curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv and install dependencies using uv
uv venv .venv
source .venv/bin/activate
uv pip install locust vllm transformers optimum auto-gptq aiohttp numpy
