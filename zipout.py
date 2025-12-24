"""
First make sure you have an `output/` directory containing:

    - evaluation logs (optional, e.g., output/evaluation_logs/)
    - plots (e.g., output/plots/, output/example_plots/)
    - evaluation_code/ (evaluation scripts)
    - README_evaluation.txt

Then run:

    python3 zipout.py

This will create a file `output.zip` in the current directory.

To customize the directory or zipfile name, run:

    python3 zipout.py -h
"""

import sys
import os
import argparse
import zipfile

def zip_dir(ziph, base_dir):
    """
    Add an entire directory tree (base_dir) into an open ZipFile.
    base_dir itself will appear as the top-level folder in the zip.
    """
    base_name = os.path.basename(base_dir.rstrip(os.sep))
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, base_dir)
            arcname = os.path.join(base_name, rel_path)
            ziph.write(file_path, arcname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--outputdir",
        dest="output_dir",
        default="output",
        help="directory that contains evaluation outputs (default: output/)"
    )

    parser.add_argument(
        "-z", "--zipfile",
        dest="zipfile",
        default="output",
        help="base name of the zip file to create (default: output → output.zip)"
    )

    opts = parser.parse_args()

    output_dir = opts.output_dir
    zip_filename = opts.zipfile + ".zip"

    if not os.path.exists(output_dir):
        print(f"Error: directory '{output_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        print(f"Zipping directory: {output_dir}")
        zip_dir(zipf, output_dir)

    print(f"{zip_filename} created", file=sys.stderr)
