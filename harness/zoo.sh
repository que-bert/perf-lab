# sourced: zoo_path <name> -> model path; zoo_names [role] -> names. Table: harness/zoo.tsv
ZOO_TSV=$(dirname "${BASH_SOURCE[0]}")/zoo.tsv
ZOO_M=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models
zoo_path() { awk -F'\t' -v n="$1" -v m="$ZOO_M" '$1==n{sub(/^M\//, m"/", $4); print $4}' "$ZOO_TSV"; }
zoo_mtp()  { awk -F'\t' -v n="$1" '$1==n{print $3}' "$ZOO_TSV"; }
zoo_names() { awk -F'\t' -v r="${1:-}" '!/^#/ && NF>=4 && (r=="" || $2==r){print $1}' "$ZOO_TSV"; }
