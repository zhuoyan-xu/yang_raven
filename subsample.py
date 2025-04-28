import json
import random
import sys

random.seed(0)

def subsample_jsonl(input_file, output_file, ratio):
    ratio = float(ratio)  # Ensure ratio is a float
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if random.random() < ratio:  # Keep ~ratio% of lines
                outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python subsample_jsonl.py <input_file> <output_file> <ratio>")
        sys.exit(1)

    input_jsonl = sys.argv[1]
    output_jsonl = sys.argv[2]
    ratio = sys.argv[3]

    subsample_jsonl(input_jsonl, output_jsonl, ratio)
    print(f"Subsampled data saved to {output_jsonl}")
