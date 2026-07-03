# Hosting a Fine-Tuned Qwen Model on a VPS using vLLM

This is a rough guide on how I'd go about deploying the fine-tuned Qwen model on a VPS using vLLM. The idea is to get an OpenAI-compatible endpoint running that can be hit from any frontend or API.

---

## 1. Pick the right VPS

For a 7B model you need at least 16GB VRAM comfortably, so something like an A10 or RTX 4090 instance works. Anything less and you'll have to run quantized (4-bit), which is fine too but adds latency. I'd go with Ubuntu 22.04 — most CUDA setups are well-documented for it.

Check the GPU is actually visible after provisioning:
```bash
nvidia-smi
```

---

## 2. Basic system setup

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git
```

If CUDA drivers aren't preinstalled by the provider (they usually are on GPU instances), install them manually. The NVIDIA docs are pretty clear on this — match the driver version to your GPU and CUDA version.

---

## 3. Set up a Python environment

Always use a venv, avoids dependency conflicts:
```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install --upgrade pip
```

---

## 4. Install vLLM

```bash
pip install vllm
```

Note — if you're using Qwen3 specifically, make sure you're on a recent enough vLLM release (0.4.x+), since Qwen3 architecture support was added later. Check the vLLM changelog to confirm.

---

## 5. Get the model onto the server

Two options here:

**Option A — Upload the merged model directly:**
```bash
scp -r ./qwen-astrologer-merged user@YOUR_VPS_IP:/home/user/models/
```

**Option B — Pull from HuggingFace Hub (Recommended):**
Since the model is on HuggingFace Hub, you can download it directly to the server:
```bash
pip install huggingface_hub
huggingface-cli download iglou/qwen-astrologer-lora --local-dir /home/user/models/qwen-astrologer-lora
```

---

## 6. Start the vLLM server

**If you merged the adapter into the base model:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/user/models/qwen-astrologer-merged \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

**If you want to serve with the LoRA adapter live (no merging):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules astrologer=/home/user/models/qwen-astrologer-lora \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16
```

The server exposes an OpenAI-compatible API, so you can call it exactly like the OpenAI SDK.

---

## 7. Keep it running with systemd

Don't leave it in a terminal tab — use systemd so it restarts on reboot or crash:

```bash
sudo nano /etc/systemd/system/vllm.service
```

Paste this in:
```ini
[Unit]
Description=vLLM Astrologer Server
After=network.target

[Service]
User=user
WorkingDirectory=/home/user
ExecStart=/home/user/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/user/models/qwen-astrologer-merged \
  --host 0.0.0.0 --port 8000 --dtype bfloat16
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
sudo systemctl status vllm
```

---

## 8. Put Nginx in front of it

Running vLLM directly on port 8000 isn't ideal for production. Set up Nginx as a reverse proxy with HTTPS:

```bash
sudo apt install nginx certbot python3-certbot-nginx -y
```

Create a site config at `/etc/nginx/sites-available/vllm`:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

Enable it and get a cert:
```bash
sudo ln -s /etc/nginx/sites-available/vllm /etc/nginx/sites-enabled/
sudo certbot --nginx -d yourdomain.com
sudo systemctl reload nginx
```

---

## 9. Add an API key

vLLM supports a simple API key out of the box:
```bash
# add to your ExecStart line in the systemd service
--api-key your-secret-key-here
```

Then restrict port 8000 to localhost only via ufw:
```bash
sudo ufw allow 22
sudo ufw allow 443
sudo ufw allow 80
sudo ufw enable
```

---

## 10. Quick test

```bash
curl https://yourdomain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key-here" \
  -d '{
    "model": "qwen-astrologer-merged",
    "messages": [
      {"role": "user", "content": "Meri shaadi kab hogi?"}
    ],
    "max_tokens": 200
  }'
```

---

## 11. Keeping an eye on it

- GPU utilization: `nvidia-smi` or `nvtop`
- Server logs: `journalctl -u vllm -f`
- Basic uptime monitoring: something like UptimeRobot pinging your endpoint every 5 minutes is enough for a start

Keep vLLM updated — the project moves fast and newer versions often fix memory issues and add model support.
