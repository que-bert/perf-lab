# sourced: zoo_path <name> -> model path; zoo_names [role] -> names. Table: harness/zoo.tsv
ZOO_TSV=$(dirname "${BASH_SOURCE[0]}")/zoo.tsv
ZOO_M=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models   # SATA: other models, KLD references (perflab-kld-ref)
ZOO_N=${PERFLAB_MODEL_DIR:-/home/bbuckham/models}        # NVMe: the 27B Q6_K / Q4_K_M, mmproj, 35B-A3B (zoo.tsv "N/" prefix)
zoo_path() { awk -F'\t' -v n="$1" -v m="$ZOO_M" -v nv="$ZOO_N" '$1==n{sub(/^M\//, m"/", $4); sub(/^N\//, nv"/", $4); print $4}' "$ZOO_TSV"; }
zoo_mtp()  { awk -F'\t' -v n="$1" '$1==n{print $3}' "$ZOO_TSV"; }
zoo_names() { awk -F'\t' -v r="${1:-}" '!/^#/ && NF>=4 && (r=="" || $2==r){print $1}' "$ZOO_TSV"; }
# lane helpers: the same card choice gpu_lock.sh makes (PERFLAB_CARD=r9700|9060), for values needed before the lock is taken
zoo_vkdev() { echo "${PERFLAB_VKDEV:-Vulkan$([ "${PERFLAB_CARD:-r9700}" = 9060 ] && echo 0 || echo 1)}"; }
zoo_port() { echo $(( $1 + $([ "${PERFLAB_CARD:-r9700}" = 9060 ] && echo 100 || echo 0) )); }
