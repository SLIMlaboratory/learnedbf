#!/bin/bash

source /opt/intel/oneapi/setvars.sh

export EPSILON=0.02

rm -f results.csv
echo "N_KEYS, N_NONKEYS, Memory_Usage_MB" > results.csv

for N_KEYS in 10000 ; do
    export N_KEYS
    export N_NONKEYS=$(( N_KEYS * 3 / 2 ))
    export N_TEST_NONKEYS=$N_NONKEYS

    export FILE_NAME="filter_${N_KEYS}_${N_NONKEYS}_${EPSILON}"
    export VTUNE_DIR="vtune-results_${N_KEYS}_${N_NONKEYS}_${EPSILON}"
    rm -rf ${VTUNE_DIR}_learned_space
    rm -rf ${VTUNE_DIR}_classical_space

    echo "Building a classical and learned Bloom filter with $N_KEYS keys and $N_NONKEYS non-keys..."
    python build_lbf.py --key_csv_path data/cicids2018_KEYS_keys_dedup.csv --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_keys $N_KEYS --n_nonkeys $N_NONKEYS --n_test_nonkeys $N_TEST_NONKEYS --file_name $FILE_NAME --epsilon $EPSILON

    echo "Testing the learned Bloom filter (Python) with $N_TEST_NONKEYS non-keys..."
    vtune -q -collect memory-consumption -result-dir ${VTUNE_DIR}_python_learned_space -- python test_lbf.py --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME 2>&1 | grep -v -e 'vtune: Warning' -e 'vtune: Executing' -e '^[[:space:]]*$'

    echo "Testing the classical Bloom filter (Python) with $N_TEST_NONKEYS non-keys..."
    vtune -collect memory-consumption -result-dir ${VTUNE_DIR}_python_classical_space -- python test_bf.py --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME  2>&1 | grep -v -e 'vtune: Warning' -e 'vtune: Executing' -e '^[[:space:]]*$'

    echo "Testing the learned Bloom filter (C) with $N_TEST_NONKEYS non-keys..."
    vtune -q -collect memory-consumption -result-dir ${VTUNE_DIR}_c_learned_space -- ./test_lbf --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME 2>&1 | grep -v -e 'vtune: Warning' -e 'vtune: Executing' -e '^[[:space:]]*$'

    echo "Testing the classical Bloom filter (C) with $N_TEST_NONKEYS non-keys..."
    vtune -q -collect memory-consumption -result-dir ${VTUNE_DIR}_c_classical_space -- ./test_bf --nonkey_csv_path data/cicids2018_NONKEYS_nonkeys_part1_dedup.csv --n_test_nonkeys $N_TEST_NONKEYS --filter_file_name $FILE_NAME 2>&1 | grep -v -e 'vtune: Warning' -e 'vtune: Executing' -e '^[[:space:]]*$'

    rm -f hotspots*.csv

    vtune -report hotspots -result-dir ${VTUNE_DIR}_python_learned_space -format csv -csv-delimiter comma -report-output hotspots_python_learned.csv
    vtune -report hotspots -result-dir ${VTUNE_DIR}_python_classical_space -format csv -csv-delimiter comma -report-output hotspots_python_classical.csv
    vtune -report hotspots -result-dir ${VTUNE_DIR}_c_learned_space -format csv -csv-delimiter comma -report-output hotspots_c_learned.csv
    vtune -report hotspots -result-dir ${VTUNE_DIR}_c_classical_space -format csv -csv-delimiter comma -report-output hotspots_c_classical.csv

    echo "python, learned, ${N_KEYS}, ${N_NONKEYS}, $(awk -F',' 'NR>1 {sum += $(NF-4)} END {printf "%.2f Kb", sum/1024}' hotspots_python_learned.csv)" >> results.csv
    echo "python, classical, ${N_KEYS}, ${N_NONKEYS}, $(awk -F',' 'NR>1 {sum += $(NF-4)} END {printf "%.2f Kb", sum/1024}' hotspots_python_classical.csv)" >> results.csv
    echo "c, learned, ${N_KEYS}, ${N_NONKEYS}, $(awk -F',' 'NR>1 {sum += $(NF-4)} END {printf "%.2f Kb", sum/1024}' hotspots_c_learned.csv)" >> results.csv
    echo "c, classical, ${N_KEYS}, ${N_NONKEYS}, $(awk -F',' 'NR>1 {sum += $(NF-4)} END {printf "%.2f Kb", sum/1024}' hotspots_c_classical.csv)" >> results.csv

    rm -rf ${VTUNE_DIR}_space

done
