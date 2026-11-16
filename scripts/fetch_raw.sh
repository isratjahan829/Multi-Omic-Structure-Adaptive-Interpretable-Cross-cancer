#!/usr/bin/env bash
# Fetch the raw cBioPortal study exports for the MOSAIC-GNN real-data run.
# Downloads land in ./raw and are treated as untrusted data (never executed).
set -u
RAW="${1:-./raw}"
mkdir -p "$RAW"

read -r -d '' FILES <<'EOF'
blca_data_clinical_patient.txt 1gZBjSzgCUjekSKKzRn7RugVYMWVBc2nB
blca_clinical_sample.txt 1CIJIRIXVkzv16IGBYzjb5FBK95rkvCv0
blca_data_mutations.txt 1GV5PLjooJ4qA56eAnxwTr3DtoTTuU_9i
blca_data_cna.txt 15a6i9y8AUzMDvC2QXtitGoOU2Vr7tXvy
blca_data_mrna_seq_v2_rsem.txt 1inPYCCVM4LBoYZwcfB_gUU7VWkSffbew
lihc_data_clinical_patient.txt 1xpo2sxaLyiGSalJcrhF5OAC5kx2r-BlR
lihc_clinical_sample.txt 1KluE6MuFSZwUgvrUeI6-24ZcLtMsMPOu
lihc_data_mutations.txt 1pG9ZZnz5jMGucfgmMgXZIwdI98p8Q472
lihc_data_cna.txt 1HoQNdu7ltxWHTlEPLA4uBWP8oUQ-dHYF
lihc_data_mrna_seq_v2_rsem.txt 1nNwJHr425XCHQ1j_jPJAwmxctyd1DW-I
thca_data_clinical_patient.txt 1d2Xz5XZeJ8A4Xt-ziK45DkrqBlF1Yye3
thca_clinical_sample.txt 1gC350jkmH54BBJ1ut_fuAmBEnlFrjQqV
thca_data_mutations.txt 11KWQmRCjSdXB5vrwx9tXc4ixfYrRhE8-
thca_data_cna.txt 1XlvHmDKBpahR3FoKQtyvuFj9OnxMrlKD
thca_data_mrna_seq_v2_rsem.txt 1MSNia2yxhTh5cEyQaUYIip7yvjvxCkTb
lung_data_clinical_patient.txt 1hD2URY7lS925ioOB9J5e6BdGzo3uELVk
lung_data_clinical_sample.txt 13xX2UMS5Glzxo9jQExeqJvMUN2A1iqZ4
lung_data_mutations.txt 1ViED1L9HybiudXN2QmTsVsRtbJCFEI34
lung_data_cna.txt 110R1XhWoRBHBgrgoCB9MDlJ_yfkXsTq3
lung_data_mrna_seq_v2_rsem.txt 1_afsZXNDl_YvaViDF6CsMJRJ7u6iM0-m
EOF

while read -r NAME ID; do
    [ -z "$NAME" ] && continue
    DEST="$RAW/$NAME"
    if [ -s "$DEST" ]; then
        echo "SKIP  $NAME (already $(stat -c%s "$DEST") bytes)"
        continue
    fi
    for attempt in 1 2 3 4; do
        code=$(curl -sSL --connect-timeout 30 --max-time 1800 \
            -w '%{http_code}' -o "$DEST" \
            "https://drive.usercontent.google.com/download?id=${ID}&export=download&confirm=t")
        size=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
        if [ "$code" = "200" ] && [ "$size" -gt 1000 ]; then
            echo "OK    $NAME  http=$code  bytes=$size"
            break
        fi
        echo "RETRY $NAME attempt=$attempt http=$code bytes=$size"
        sleep $((attempt * 10))
    done
done <<< "$FILES"

echo "=== download pass finished ==="
