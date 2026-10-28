#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
workspace="$(cd "$root/../.." && pwd)"
cache="$workspace/runs/single_paper_20261028/cache"
mkdir -p "$cache"
args=(--die-with-parent --new-session --unshare-net --unshare-ipc --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp
 --bind "$workspace" "$workspace" --ro-bind "$workspace/repos/ARES" "$workspace/repos/ARES"
 --chdir "$root" --setenv CUDA_VISIBLE_DEVICES 0 --setenv OMP_NUM_THREADS 4 --setenv OPENBLAS_NUM_THREADS 4
 --setenv XDG_CACHE_HOME "$cache" --setenv MPLCONFIGDIR "$cache/matplotlib"
 --setenv VK_ICD_FILENAMES /usr/share/vulkan/icd.d/nvidia_icd.json
 --setenv PYTHONPATH "$workspace/external/benchmark_scout_20261027/ManiSkill:$root"
 --setenv PYTHONUNBUFFERED 1 --setenv DISPLAY "")
for prior in "$workspace/worktrees/ares-sci-method-evolution-20261028"; do
 [[ ! -d "$prior" ]] || args+=(--ro-bind "$prior" "$prior")
done
# Protect every ARES-R checkout, including historical presentation worktrees.
while IFS= read -r checkout; do
 [[ ! -d "$checkout" ]] || args+=(--ro-bind "$checkout" "$checkout")
done < <(git -C "$workspace/repos/ARES-R" worktree list --porcelain | sed -n 's/^worktree //p')
for device in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
 [[ ! -e "$device" ]] || args+=(--dev-bind "$device" "$device")
done
[[ ! -d /dev/dri ]] || args+=(--dev-bind /dev/dri /dev/dri)
exec bwrap "${args[@]}" -- "$@"
