#!/bin/bash
# Bring a fresh RunPod worker online: ./setup_pod.sh <ssh_port> <shard K/N> <logname>
# e.g. ./setup_pod.sh 14200 0/6 worker_p7.log
set -e
PORT=$1; SHARD=$2; LOG=$3
HOST=root@209.170.80.132
SSH="ssh -o StrictHostKeyChecking=accept-new -p $PORT $HOST"

$SSH 'apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq poppler-utils > /dev/null 2>&1; which pdftoppm' | grep -q pdftoppm || { echo "POPPLER INSTALL FAILED"; exit 1; }
$SSH "nvidia-smi --query-gpu=name --format=csv,noheader"
$SSH "cd /workspace && setsid nohup venv/bin/python cloud_ocr.py --files ocr_full_list_v2.txt --shard $SHARD > $LOG 2>&1 < /dev/null & disown; echo launched"
sleep 8
$SSH "head -1 /workspace/$LOG"
