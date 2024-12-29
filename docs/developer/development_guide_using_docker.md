# Development Guide Using Docker

## Setup VSCode

Download `code` from `Https://code.visualstudio.com/docs/?dv=linux64cli`

```bash
wget https://vscode.download.prss.microsoft.com/dbazure/download/stable/fabdb6a30b49f79a7aba0f2ad9df9b399473380f/vscode_cli_alpine_x64_cli.tar.gz
tar xf vscode_cli_alpine_x64_cli.tar.gz

# https://code.visualstudio.com/docs/remote/tunnels
./code tunnel
```

## Setup Docker Container

### H100

```bash
# Change the name to yours
docker run -itd --shm-size 32g --gpus all -v /opt/dlami/nvme/.cache:/root/.cache --ipc=host --name sglang_zhyncs lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_zhyncs /bin/zsh
```

### H200

```bash
docker run -itd --shm-size 32g --gpus all -v /mnt/co-research/shared-models:/root/.cache/huggingface --ipc=host --name sglang_zhyncs lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_zhyncs /bin/zsh
```

## Profile

```bash
# Change batch size, input, output and add `disable-cuda-graph` (for easier analysis)
# e.g. DeepSeek V3
nsys profile -o deepseek_v3 python3 -m sglang.bench_one_batch --batch-size 1 --input 128 --output 256 --model deepseek-ai/DeepSeek-V3 --trust-remote-code --tp 8 --disable-cuda-graph
```

## Evaluation

```bash
# e.g. gsm8k 8 shot
python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8
```