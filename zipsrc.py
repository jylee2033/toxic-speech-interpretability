"""
Run:

    python3 zipsrc.py

This will create a file `source.zip` containing your source code submission
for Coursys (courses.cs.sfu.ca).

To customize the files or folders included in the archive, run:

    python3 zipsrc.py -h
"""

import sys, os, argparse, shutil, zipfile

def zipdir(ziph, folder):
    """Zip an entire directory tree."""
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath)
            ziph.write(filepath, arcname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-z", "--zipfile",
        dest="zipfile",
        default="source",
        help="name of zip file to create (default: source)"
    )

    # Source code folders to include
    parser.add_argument(
        "-f", "--folders",
        nargs="*",
        default=["preprocessing", "models", "evaluation", "interpretability"],
        help="folders containing source code to include"
    )

    # Individual files to include (write-up, requirements, README files, run_pipeline)
    parser.add_argument(
        "-i", "--include",
        nargs="*",
        default=[
            "project.ipynb",
            "requirements.txt",
            "README.jaa60",
            "README.jla670",
            "run_pipeline.py"
        ],
        help="list of standalone files to include"
    )

    opts = parser.parse_args()

    zip_filename = opts.zipfile + ".zip"

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:

        # Add directories
        for folder in opts.folders:
            if os.path.exists(folder):
                print(f"Adding folder: {folder}")
                zipdir(zipf, folder)
            else:
                print(f"Warning: folder not found: {folder}")

        # Add individual files
        for file in opts.include:
            if os.path.exists(file):
                print(f"Adding file: {file}")
                zipf.write(file, file)
            else:
                print(f"Warning: file not found: {file}")

    print(f"\nCreated {zip_filename} successfully!", file=sys.stderr)
