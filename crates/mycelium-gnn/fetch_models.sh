#!/usr/bin/env bash
# Fetch pretrained models, tokenizers, and GloVe vectors.
#
# These files are too large for git — run this once after cloning.
#
# Downloads:
#   models/model.onnx              ~90 MB  sentence-transformers/all-MiniLM-L6-v2
#   models/tokenizer.json          ~466 KB
#   models/cross-encoder.onnx      ~90 MB  cross-encoder/ms-marco-MiniLM-L-6-v2
#   models/cross-tokenizer.json    ~711 KB
#   demo/glove.6B.50d.txt          ~171 MB GloVe 6B 50d vectors
#
# Usage:
#   cd crates/mycelium-gnn && ./fetch_models.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
DEMO_DIR="$SCRIPT_DIR/demo"

HF="https://huggingface.co"
BI_ENCODER="sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER="cross-encoder/ms-marco-MiniLM-L-6-v2"

mkdir -p "$MODEL_DIR" "$DEMO_DIR"

fetch() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        echo "  skip  $(basename "$dest") (exists)"
        return
    fi
    echo "  fetch $(basename "$dest")"
    wget -q --show-progress -O "$dest" "$url"
}

# ---------------------------------------------------------------------------
# Bi-encoder: sentence-transformers/all-MiniLM-L6-v2
# ---------------------------------------------------------------------------
echo "=== Bi-encoder (MiniLM) ==="
fetch "$HF/$BI_ENCODER/resolve/main/onnx/model.onnx" \
      "$MODEL_DIR/model.onnx"
fetch "$HF/$BI_ENCODER/resolve/main/tokenizer.json" \
      "$MODEL_DIR/tokenizer.json"

# ---------------------------------------------------------------------------
# Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
# ---------------------------------------------------------------------------
echo "=== Cross-encoder (ms-marco-MiniLM) ==="
fetch "$HF/$CROSS_ENCODER/resolve/main/onnx/model.onnx" \
      "$MODEL_DIR/cross-encoder.onnx"
fetch "$HF/$CROSS_ENCODER/resolve/main/tokenizer.json" \
      "$MODEL_DIR/cross-tokenizer.json"

# ---------------------------------------------------------------------------
# GloVe 6B 50d
# ---------------------------------------------------------------------------
echo "=== GloVe 6B 50d ==="
if [ -f "$DEMO_DIR/glove.6B.50d.txt" ]; then
    echo "  skip  glove.6B.50d.txt (exists)"
else
    GLOVE_ZIP="$DEMO_DIR/glove.6B.zip"
    echo "  fetch glove.6B.zip (~862 MB)"
    wget -q --show-progress -O "$GLOVE_ZIP" \
        "https://nlp.stanford.edu/data/glove.6B.zip"
    echo "  extract glove.6B.50d.txt"
    unzip -j -o "$GLOVE_ZIP" "glove.6B.50d.txt" -d "$DEMO_DIR"
    rm -f "$GLOVE_ZIP"
fi

echo ""
echo "done. files:"
ls -lh "$MODEL_DIR"
ls -lh "$DEMO_DIR/glove.6B.50d.txt"
