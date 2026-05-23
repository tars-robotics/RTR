"""
python src/rtr_async_sys/tools/count_lines.py src/rtr_async_sys src/rtr_async_sys/models
"""
import os
import sys

def is_python_file(path):
    return path.endswith(".py")

def count_code_lines(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            # Exclude blank and comment lines
            if stripped and not stripped.startswith("#"):
                count += 1
    return count

def count_project_lines(target_dir, exclude_dir):
    total_lines = 0
    file_count = 0

    for root, dirs, files in os.walk(target_dir):
        # Skip excluded directories
        if exclude_dir and os.path.abspath(root).startswith(os.path.abspath(exclude_dir)):
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if is_python_file(file_path):
                file_count += 1
                lines = count_code_lines(file_path)
                total_lines += lines
                print(f"{file_path}: {lines} lines")

    print("\n==========================")
    print(f"Total Python files: {file_count}")
    print(f"Total code lines (excluding empty/comment lines): {total_lines}")
    print("==========================")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python count_lines.py <target_dir> <exclude_dir>")
        sys.exit(1)

    target_dir = sys.argv[1]
    exclude_dir = sys.argv[2]

    print(f"Counting .py lines in: {target_dir}")
    print(f"Excluding directory: {exclude_dir}")
    count_project_lines(target_dir, exclude_dir)
