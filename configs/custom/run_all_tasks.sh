#!/usr/bin/env bash

# Run complete 4-Task Continual Learning for IP102 (25 classes total)
set -e

echo "=== 1. Preparing Dataset Splits (Task 1: 7 classes, Task 2-4: 6 classes each) ==="
python configs/custom/prepare_data_splits.py

echo "=== 2. Running Task 1 (Classes 0..6) ==="
bash configs/custom/ip102_t1.sh

echo "=== 3. Running Task 2 (Classes 7..12) ==="
bash configs/custom/ip102_t2.sh

echo "=== 4. Running Task 3 (Classes 13..18) ==="
bash configs/custom/ip102_t3.sh

echo "=== 5. Running Task 4 (Classes 19..24) ==="
bash configs/custom/ip102_t4.sh

echo "=== All 4 Tasks Completed Successfully! ==="
