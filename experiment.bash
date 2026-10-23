#!/bin/bash

MEASUREMENT=${MEASUREMENT:-${1:-time}}
case "$MEASUREMENT" in
    time|vtune|both) ;;
    *) echo "Usage: $0 [time|vtune|both]" >&2; exit 2 ;;
esac

PYTHON=${PYTHON:-python}
if [[ "$PYTHON" == "python" && -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
fi

if [[ "$MEASUREMENT" == "vtune" || "$MEASUREMENT" == "both" ]]; then
    source /opt/intel/oneapi/setvars.sh
fi

export EPSILON=0.01

rm -f results.csv
echo "Language, Filter, N_KEYS, N_NONKEYS, Filter_Size_KB, Peak_RSS_KB, VTune_Net_Allocation_Delta_KB" > results.csv

for N_KEYS in 100000 200000 300000 400000 500000 600000 ; do
    export N_KEYS
    export N_NONKEYS=$(( N_KEYS * 3 / 2 ))
    export N_TEST_NONKEYS=$N_NONKEYS

    export FILE_NAME="filter_${N_KEYS}_${N_NONKEYS}_${EPSILON}"

    echo "Building a classical and learned Bloom filter with $N_KEYS keys and $N_NONKEYS non-keys..."
    "$PYTHON" build_lbf.py --key_csv_path data/cicids2018_KEYS_keys_dedup.csv --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_keys $N_KEYS --n_nonkeys $N_NONKEYS --n_test_nonkeys $N_TEST_NONKEYS --file_name $FILE_NAME --epsilon $EPSILON

    echo "Building C test programs..."
    gcc -std=c11 -O2 -Wall -Wextra -Wpedantic test_lbf.c -lm -o test_lbf
    gcc -std=c11 -O2 -Wall -Wextra -Wpedantic test_bf.c -lm -o test_bf

    PEAK_RSS_PYTHON_LEARNED_KB=""
    PEAK_RSS_PYTHON_CLASSICAL_KB=""
    PEAK_RSS_C_LEARNED_KB=""
    PEAK_RSS_C_CLASSICAL_KB=""
    VTUNE_PYTHON_LEARNED_KB=""
    VTUNE_PYTHON_CLASSICAL_KB=""
    VTUNE_C_LEARNED_KB=""
    VTUNE_C_CLASSICAL_KB=""

    if [[ "$MEASUREMENT" == "time" || "$MEASUREMENT" == "both" ]]; then
        echo "Measuring peak RSS with GNU time..."
        /usr/bin/time -f '%M' -o "${FILE_NAME}.python_learned.peak_kib" "$PYTHON" test_lbf.py --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME
        /usr/bin/time -f '%M' -o "${FILE_NAME}.python_classical.peak_kib" "$PYTHON" test_bf.py --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME
        /usr/bin/time -f '%M' -o "${FILE_NAME}.c_learned.peak_kib" ./test_lbf --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME
        /usr/bin/time -f '%M' -o "${FILE_NAME}.c_classical.peak_kib" ./test_bf --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME

        PEAK_RSS_PYTHON_LEARNED_KB=$(awk '{printf "%.2f", $1 * 1.024}' "${FILE_NAME}.python_learned.peak_kib")
        PEAK_RSS_PYTHON_CLASSICAL_KB=$(awk '{printf "%.2f", $1 * 1.024}' "${FILE_NAME}.python_classical.peak_kib")
        PEAK_RSS_C_LEARNED_KB=$(awk '{printf "%.2f", $1 * 1.024}' "${FILE_NAME}.c_learned.peak_kib")
        PEAK_RSS_C_CLASSICAL_KB=$(awk '{printf "%.2f", $1 * 1.024}' "${FILE_NAME}.c_classical.peak_kib")
    fi

    if [[ "$MEASUREMENT" == "vtune" || "$MEASUREMENT" == "both" ]]; then
        VTUNE_DIR="vtune-results_${N_KEYS}_${N_NONKEYS}_${EPSILON}"
        rm -rf "${VTUNE_DIR}_python_learned_space" "${VTUNE_DIR}_python_classical_space" \
               "${VTUNE_DIR}_c_learned_space" "${VTUNE_DIR}_c_classical_space"

        echo "Collecting VTune memory-consumption results..."
        vtune -q -collect memory-consumption -result-dir "${VTUNE_DIR}_python_learned_space" -- "$PYTHON" test_lbf.py --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME
        vtune -q -collect memory-consumption -result-dir "${VTUNE_DIR}_python_classical_space" -- "$PYTHON" test_bf.py --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME
        vtune -q -collect memory-consumption -result-dir "${VTUNE_DIR}_c_learned_space" -- ./test_lbf --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME
        vtune -q -collect memory-consumption -result-dir "${VTUNE_DIR}_c_classical_space" -- ./test_bf --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME

        vtune -report hotspots -result-dir "${VTUNE_DIR}_python_learned_space" -format csv -csv-delimiter comma -report-output hotspots_python_learned.csv
        vtune -report hotspots -result-dir "${VTUNE_DIR}_python_classical_space" -format csv -csv-delimiter comma -report-output hotspots_python_classical.csv
        vtune -report hotspots -result-dir "${VTUNE_DIR}_c_learned_space" -format csv -csv-delimiter comma -report-output hotspots_c_learned.csv
        vtune -report hotspots -result-dir "${VTUNE_DIR}_c_classical_space" -format csv -csv-delimiter comma -report-output hotspots_c_classical.csv

        VTUNE_PYTHON_LEARNED_KB=$(awk -F',' 'NR>1 {sum += $2} END {printf "%.2f", sum/1000}' hotspots_python_learned.csv)
        VTUNE_PYTHON_CLASSICAL_KB=$(awk -F',' 'NR>1 {sum += $2} END {printf "%.2f", sum/1000}' hotspots_python_classical.csv)
        VTUNE_C_LEARNED_KB=$(awk -F',' 'NR>1 {sum += $2} END {printf "%.2f", sum/1000}' hotspots_c_learned.csv)
        VTUNE_C_CLASSICAL_KB=$(awk -F',' 'NR>1 {sum += $2} END {printf "%.2f", sum/1000}' hotspots_c_classical.csv)
    fi

    LEARNED_SIZE_KB=$(awk '$1 == "size:" {printf "%.2f", $2 / 8000}' "${FILE_NAME}.learned.fpr")
    CLASSICAL_SIZE_KB=$(awk '$1 == "size:" {printf "%.2f", $2 / 8000}' "${FILE_NAME}.classical.fpr")
    echo "python, learned, ${N_KEYS}, ${N_NONKEYS}, ${LEARNED_SIZE_KB}, ${PEAK_RSS_PYTHON_LEARNED_KB}, ${VTUNE_PYTHON_LEARNED_KB}" >> results.csv
    echo "python, classical, ${N_KEYS}, ${N_NONKEYS}, ${CLASSICAL_SIZE_KB}, ${PEAK_RSS_PYTHON_CLASSICAL_KB}, ${VTUNE_PYTHON_CLASSICAL_KB}" >> results.csv
    echo "c, learned, ${N_KEYS}, ${N_NONKEYS}, ${LEARNED_SIZE_KB}, ${PEAK_RSS_C_LEARNED_KB}, ${VTUNE_C_LEARNED_KB}" >> results.csv
    echo "c, classical, ${N_KEYS}, ${N_NONKEYS}, ${CLASSICAL_SIZE_KB}, ${PEAK_RSS_C_CLASSICAL_KB}, ${VTUNE_C_CLASSICAL_KB}" >> results.csv

done
