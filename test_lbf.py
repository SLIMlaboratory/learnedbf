from learnedbf import LBF
import numpy as np

if __name__ == "__main__":
    # using argc and argv, fecth csv paths and other params from command line
    import argparse
    
    # first argument is key csv path, second argument is non-key csv path,
    # third argument is n_keys, fourth argument is n_nonkeys
    parser = argparse.ArgumentParser(description="Evaluate a learned"
                                     " Bloom filter from CSV data.")

    parser.add_argument("--nonkey_csv_path", type=str,
                            help="Path to the CSV file containing non-key rows.")
    parser.add_argument("--n_test_nonkeys", type=int,
                        help="Number of non-keys to test against the Bloom filter.")
    parser.add_argument("--filter_file_name", type=str,
                        help="Filter file name.")

    args = parser.parse_args()

    learned_filter = LBF()
    learned_filter.import_(f"{args.filter_file_name}.learned.json")
    
    with open(args.nonkey_csv_path, "rb") as f:
        # read one nonkey at a time and probe the learned filter, counting false positives
        false_positives = 0
        total_probed = 0

        next(f).decode().strip("\n").strip("\r") # skip header
        while total_probed < args.n_test_nonkeys:
            row = next(f).decode().strip("\n").strip("\r")
            nonkey = [np.float32(x) for x in row.split(",")]

            if learned_filter.predict([nonkey])[0] == True:
                false_positives += 1
            total_probed += 1
    
    fpr = false_positives / total_probed
    print(f"False positive rate: {fpr:.6f}.")

    size = sum(learned_filter.get_size().values())
    print(f"Size of the learned Bloom filter: {size} bits.")


    
