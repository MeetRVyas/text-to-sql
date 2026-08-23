#!/usr/bin/env bash
# scripts/download_spider.sh
# --------------------------
# Downloads the Spider Text-to-SQL dataset and extracts it to data/spider/.
#
# Spider is hosted on Yale's servers and requires accepting a usage agreement
# on https://yale-lily.github.io/spider before downloading.
#
# Usage:
#   bash scripts/download_spider.sh
#
# After running this script the layout should be:
#   data/spider/
#     train_spider.json
#     dev.json
#     tables.json
#     database/
#       concert_singer/
#         concert_singer.db
#       ...

set -euo pipefail

SPIDER_URL="https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
DEST_DIR="data"
ZIP_NAME="spider.zip"

echo "=================================================="
echo " Spider Text-to-SQL dataset downloader"
echo "=================================================="
echo ""
echo "NOTE: By downloading Spider you agree to the Yale"
echo "      non-commercial research licence."
echo "      See https://yale-lily.github.io/spider"
echo ""

mkdir -p "$DEST_DIR"

if [ -d "$DEST_DIR/spider" ]; then
    echo "[skip] data/spider already exists. Delete it to re-download."
    exit 0
fi

echo "[1/3] Downloading Spider …"
# Use wget with progress bar; fall back to curl
if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$DEST_DIR/$ZIP_NAME" "$SPIDER_URL"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$DEST_DIR/$ZIP_NAME" "$SPIDER_URL"
else
    echo "ERROR: neither wget nor curl found. Install one and retry."
    exit 1
fi

echo "[2/3] Extracting …"
unzip -q "$DEST_DIR/$ZIP_NAME" -d "$DEST_DIR"
rm "$DEST_DIR/$ZIP_NAME"

echo "[3/3] Verifying layout …"
required_files=(
    "$DEST_DIR/spider/train_spider.json"
    "$DEST_DIR/spider/dev.json"
    "$DEST_DIR/spider/tables.json"
    "$DEST_DIR/spider/database"
)

all_ok=true
for f in "${required_files[@]}"; do
    if [ ! -e "$f" ]; then
        echo "  MISSING: $f"
        all_ok=false
    else
        echo "  OK: $f"
    fi
done

if $all_ok; then
    echo ""
    echo "Spider dataset ready at data/spider/"
    db_count=$(find data/spider/database -name "*.db" | wc -l)
    echo "  Found $db_count .db files."
else
    echo ""
    echo "WARNING: Some expected files are missing. The zip structure may have"
    echo "changed. Check the contents of $DEST_DIR and adjust paths if needed."
fi
