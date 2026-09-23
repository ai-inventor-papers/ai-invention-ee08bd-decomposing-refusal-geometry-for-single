#!/usr/bin/env bash
# Restore model weights deleted after the round (see README "Restoring removed files").
set -x
for repo in \
  Qwen/Qwen3-4B-Base Qwen/Qwen3-4B Qwen/Qwen3-4B-SafeRL mlabonne/Qwen3-4B-abliterated \
  Qwen/Qwen3-4B-Instruct-2507 huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated \
  Qwen/Qwen3-1.7B-Base Qwen/Qwen3-1.7B Qwen/Qwen2.5-3B-Instruct \
  huihui-ai/Qwen2.5-3B-Instruct-abliterated-SFT; do
  huggingface-cli download "$repo"
done
