#!/usr/bin/env bash
set -eEo pipefail

repo_root=$(git rev-parse --show-toplevel)
treeish=${1:-HEAD}
label=${2:-$(git rev-parse --short "$treeish")}
tree_id=$(git rev-parse "$treeish")
object_type=$(git cat-file -t "$tree_id")
destination="$repo_root/data/review_60fd/isolated_tree_${label}"
log_dir="$repo_root/artifacts/review_60fd"

if [[ -e "$destination" ]]; then
  echo "refusing to overwrite retained isolated source: $destination" >&2
  exit 2
fi

mkdir -p "$destination" "$log_dir"
git -C "$repo_root" archive "$tree_id" | tar -x -C "$destination" \
  --exclude='artifacts/trace2decision_i3/output/test.jsonl'

prohibited=(
  'artifacts/trace2decision_i3/output/test.jsonl'
  'artifacts/experiment_i3/preflight/runs'
  'release/i3/bundle/model/model.safetensors'
  'data/cache'
)
for relative in "${prohibited[@]}"; do
  if [[ -e "$destination/$relative" ]]; then
    echo "isolated source contains prohibited prerequisite: $relative" >&2
    exit 3
  fi
done

inventory="$log_dir/isolated_tree_${label}_inventory.txt"
{
  echo "tree_id=$tree_id"
  echo "object_type=$object_type"
  echo "source_snapshot=${treeish}"
  echo "destination=${destination#$repo_root/}"
  for relative in "${prohibited[@]}"; do
    echo "absent=$relative"
  done
  find "$destination/artifacts/review_60fd/dev_inputs" -maxdepth 1 -type f -printf 'input=%P %s bytes\n' | sort
  find "$destination/training/laya_trace_i3" -maxdepth 1 -type f -printf 'source=%P %s bytes\n' | sort
  find "$destination/release/i3" -maxdepth 1 -type f -printf 'source=%P %s bytes\n' | sort
} > "$inventory"

run_suite() {
  local log=$1
  (
    cd "$destination"
    env \
      CUDA_VISIBLE_DEVICES='' \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      "$repo_root/.venv/bin/python" -B -m unittest \
      training.laya_trace_i3.test_i3 \
      training.laya_trace_i3.test_orchestration \
      scripts.test_i3_dev_diagnostics \
      release.i3.test_inference
  ) 2>&1 | tee "$log"
}

run_suite "$log_dir/isolated_tree_${label}_run1.log"
run_suite "$log_dir/isolated_tree_${label}_run2.log"

(
  cd "$destination"
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    "$repo_root/.venv/bin/python" -B scripts/i3_dev_diagnostics.py
) 2>&1 | tee "$log_dir/isolated_tree_${label}_diagnostics.log"
(
  cd "$destination"
  env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    "$repo_root/.venv/bin/python" -B scripts/i3_evidence_audit.py --dev-only
) 2>&1 | tee "$log_dir/isolated_tree_${label}_dev_audit.log"

package_parity_log="$log_dir/isolated_tree_${label}_package_parity_unavailable.log"
if (
  cd "$destination"
  env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    "$repo_root/.venv/bin/python" -B scripts/i3_verify_package.py
) > "$package_parity_log" 2>&1; then
  echo "package parity unexpectedly ran without local model weights" >&2
  exit 4
fi
if ! grep -q 'PACKAGE PARITY unavailable; missing local prerequisites:.*release/i3/bundle/model/model.safetensors' "$package_parity_log"; then
  cat "$package_parity_log" >&2
  echo "package parity did not fail closed with the expected missing-weight prerequisite" >&2
  exit 5
fi

echo "isolated CPU suite passed twice"
echo "source_object_type=$object_type"
echo "inventory=${inventory#$repo_root/}"
echo "run1=${log_dir#$repo_root/}/isolated_tree_${label}_run1.log"
echo "run2=${log_dir#$repo_root/}/isolated_tree_${label}_run2.log"
echo "diagnostics=${log_dir#$repo_root/}/isolated_tree_${label}_diagnostics.log"
echo "dev_audit=${log_dir#$repo_root/}/isolated_tree_${label}_dev_audit.log"
echo "package_parity_prerequisite=${log_dir#$repo_root/}/isolated_tree_${label}_package_parity_unavailable.log"
