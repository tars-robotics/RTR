import os
import subprocess

def find_mp4_files(directory):
    """Recursively find all MP4 files in the specified directory and its subdirectories"""
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

def reencode_mp4(input_file, output_file):
    """Re-encode MP4 files using FFmpeg"""
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'medium',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_file
    ]
    subprocess.run(command, check=True)

def replace_with_reencoded_mp4(directory):
    """Traverse all MP4 files in the directory and its subdirectories, and replace the original files with re-encoded files"""
    mp4_files = find_mp4_files(directory)
    for input_file in mp4_files:
        # build output filename
        output_file = input_file.rsplit('.', 1)[0] + '_reencoded.mp4'
        # re-encode the file
        reencode_mp4(input_file, output_file)
        # remove the original
        os.remove(input_file)
        # rename re-encoded file to the original filename
        os.rename(output_file, input_file)
        print(f"Reencoded and replaced: {input_file}")

if __name__ == "__main__":
    directory = ""  # Replace with your directory path
    replace_with_reencoded_mp4(directory)