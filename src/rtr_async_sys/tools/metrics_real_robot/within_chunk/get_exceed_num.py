import os
import argparse

def count_execute_exceed_in_file(filepath: str) -> int:
    """Count occurrences of 'execute exceed' in a single print_xxx.txt."""
    count = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "execute exceed" in line:
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Count 'execute exceed' occurrences in print_xxx.txt files.")
    parser.add_argument(
        "--traj_dir",
        type=str,
        required=True,
        help="Directory containing print_xxx.txt files"
    )
    args = parser.parse_args()

    traj_dir = args.traj_dir

    if not os.path.isdir(traj_dir):
        raise NotADirectoryError(f"{traj_dir} is not a valid directory")

    # Find print_xxx.txt
    print_files = sorted([
        os.path.join(traj_dir, f)
        for f in os.listdir(traj_dir)
        if f.startswith("print_") and f.endswith(".txt")
    ])

    if len(print_files) == 0:
        print("No print_xxx.txt files found.")
        return

    counts = []
    for pf in print_files:
        c = count_execute_exceed_in_file(pf)
        counts.append(c)
        print(f"{os.path.basename(pf)}: {c}")

    avg = sum(counts) / len(counts)

    print("=== Summary ===")
    print(f"num_files: {len(print_files)}")
    print(f"total execute exceed: {sum(counts)}")
    print(f"average per file: {avg:.6g}\n\n")

if __name__ == "__main__":
    main()
