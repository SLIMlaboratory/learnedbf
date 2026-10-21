#!/bin/bash

source /opt/intel/oneapi/setvars.sh

export EPSILON=0.001

export N_NONKEYS_TEST=10000

rm -f results*.csv
echo "N_KEYS, N_NONKEYS, Memory_Usage_MB" > results.csv

for N_KEYS in 1000 ; do
    export N_KEYS
    export N_NONKEYS=$(( N_KEYS * 3 / 2 ))
    export N_TEST_NONKEYS=$N_NONKEYS

    export FILE_NAME="filter_${N_KEYS}_${N_NONKEYS}_${EPSILON}"
    export VTUNE_DIR="vtune-results_${N_KEYS}_${N_NONKEYS}_${EPSILON}"
    rm -rf ${VTUNE_DIR}_learned_space
    rm -rf ${VTUNE_DIR}_classical_space

    echo "Building a classical and learned Bloom filter with $N_KEYS keys and $N_NONKEYS non-keys..."
    python build_lbf.py --key_csv_path data/cicids2018_KEYS_keys_dedup.csv --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_keys $N_KEYS --n_nonkeys $N_NONKEYS --n_test_nonkeys $N_TEST_NONKEYS --file_name $FILE_NAME --epsilon $EPSILON

    echo "Testing the learned Bloom filter with $N_NONKEYS_TEST non-keys..."
    vtune -collect memory-consumption -result-dir ${VTUNE_DIR}_learned_space -- python test_lbf.py --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME

    echo "Testing the classical Bloom filter with $N_NONKEYS_TEST non-keys..."
    vtune -collect memory-consumption -result-dir ${VTUNE_DIR}_classical_space -- python test_bf.py --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME

    rm -f hotspots*.csv

    vtune -report hotspots -result-dir ${VTUNE_DIR}_learned_space -format csv -csv-delimiter comma -report-output hotspots_learned.csv
    vtune -report hotspots -result-dir ${VTUNE_DIR}_classical_space -format csv -csv-delimiter comma -report-output hotspots_classical.csv

    echo "${N_KEYS}, ${N_NONKEYS}, $(awk -F',' 'NR>1 {sum += $(NF-4)} END {printf "%.2f Kb", sum/1024}' hotspots_learned.csv)" >> results_learned.csv
    echo "${N_KEYS}, ${N_NONKEYS}, $(awk -F',' 'NR>1 {sum += $(NF-4)} END {printf "%.2f Kb", sum/1024}' hotspots_classical.csv)" >> results_classical.csv

    rm -rf ${VTUNE_DIR}_space

done
