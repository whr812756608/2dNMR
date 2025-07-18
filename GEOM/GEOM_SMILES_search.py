import subprocess
import re
import argparse
import os
import sys
import shutil
import tempfile


def extract_specific_smiles(tar_file, target_smiles, output_dir=None):
    """
    Extract a specific SMILES pickle file from the tar archive.
    Searches in both rdkit_folder/qm9/ and rdkit_folder/drugs/ directories.
    Handles case variations since filenames may have mixed case but input is always uppercase.

    Args:
        tar_file (str): Path to the tar.gz file
        target_smiles (str): SMILES string to search for (always uppercase input)
        output_dir (str): Directory to extract to (optional)

    Returns:
        list: List of extracted file paths
    """
    print(f"Searching for SMILES: {target_smiles} (input is uppercase)")
    print(f"In tar file: {tar_file}")

    # Check if tar file exists
    if not os.path.exists(tar_file):
        print(f"ERROR: Tar file not found: {tar_file}")
        return []

    try:
        # Get all files from tar
        print("Listing files in tar archive...")
        result = subprocess.run(['tar', '-tzf', tar_file],
                                capture_output=True, text=True, check=True)
        files = result.stdout.strip().split('\n')

        # Filter for both qm9 and drugs pickle files
        qm9_files = [f for f in files if 'rdkit_folder/qm9/' in f and f.endswith('.pickle')]
        drugs_files = [f for f in files if 'rdkit_folder/drugs/' in f and f.endswith('.pickle')]

        all_pickle_files = qm9_files + drugs_files

        print(f"Found {len(qm9_files)} QM9 pickle files in archive")
        print(f"Found {len(drugs_files)} Drugs pickle files in archive")
        print(f"Total pickle files: {len(all_pickle_files)}")

        if not all_pickle_files:
            print("ERROR: No QM9 or Drugs pickle files found in the archive")
            return []

        # Create different case variations to try
        search_variations = [
            target_smiles,  # Original uppercase
            target_smiles.lower(),  # All lowercase
        ]

        # Search for matches with different case combinations
        matches = []
        print(f"\nSearching for case variations...")

        for variation in search_variations:
            print(f"   Trying: {variation}")
            for file_path in all_pickle_files:
                filename = file_path.split('/')[-1].replace('.pickle', '')
                source_type = "QM9" if 'rdkit_folder/qm9/' in file_path else "Drugs"

                # Check for exact match with current variation
                if filename == variation:
                    matches.append(file_path)
                    print(f"Found exact match in {source_type}: {filename}")
                    break

            if matches:
                break  # Stop searching once we find a match

        # If no exact matches, try case-insensitive search
        if not matches:
            print(f"   Trying case-insensitive search...")
            target_lower = target_smiles.lower()

            for file_path in all_pickle_files:
                filename = file_path.split('/')[-1].replace('.pickle', '')
                source_type = "QM9" if 'rdkit_folder/qm9/' in file_path else "Drugs"

                # Case-insensitive comparison
                if filename.lower() == target_lower:
                    matches.append(file_path)
                    print(f"Found case-insensitive match in {source_type}: {filename}")
                    break

        if not matches:
            print(f"ERROR: No matches found for SMILES: {target_smiles}")
            print(f"   Tried variations: {search_variations}")

            # Show some sample filenames to help debug
            print(f"\nSample QM9 filenames (first 5):")
            for i, file_path in enumerate(qm9_files[:5]):
                filename = file_path.split('/')[-1].replace('.pickle', '')
                print(f"   {i + 1}. {filename}")

            print(f"\nSample Drugs filenames (first 5):")
            for i, file_path in enumerate(drugs_files[:5]):
                filename = file_path.split('/')[-1].replace('.pickle', '')
                print(f"   {i + 1}. {filename}")
            return []

        # Extract the found files
        extracted_files = []
        print(f"\nExtracting {len(matches)} matching file(s)...")

        for match in matches:
            try:
                source_type = "QM9" if 'rdkit_folder/qm9/' in match else "Drugs"

                # Create destination directory first
                if output_dir:
                    dest_dir = os.path.join(output_dir, os.path.dirname(match))
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_file = os.path.join(output_dir, match)
                else:
                    dest_dir = os.path.dirname(match)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_file = match

                # Method 1: Try using Python's tarfile module (handles special characters better)
                try:
                    import tarfile
                    print(f"   Trying Python tarfile extraction...")

                    with tarfile.open(tar_file, 'r') as tar:
                        # Extract the specific file
                        member = tar.getmember(match)
                        extracted_file = tar.extractfile(member)

                        if extracted_file:
                            # Write to destination
                            with open(dest_file, 'wb') as f:
                                f.write(extracted_file.read())

                            extracted_files.append(dest_file)
                            filename = match.split('/')[-1].replace('.pickle', '')
                            print(f"Extracted from {source_type}: {filename}")
                            continue  # Success, move to next file

                except Exception as tarfile_error:
                    print(f"   Python tarfile method failed: {tarfile_error}")

                # Method 2: Try tar command with proper escaping
                try:
                    print(f"   Trying tar command with escaping...")

                    # Create a temporary file list to avoid command line escaping issues
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt',
                                                     encoding='utf-8') as tmp_file:
                        tmp_file.write(match + '\n')
                        tmp_file_path = tmp_file.name

                    # Extract using file list
                    extract_cmd = ['tar', '-xzf', tar_file, '--files-from', tmp_file_path]
                    if output_dir:
                        extract_cmd.extend(['-C', output_dir])

                    result = subprocess.run(extract_cmd, check=True, capture_output=True, text=True)

                    # Clean up temp file
                    os.unlink(tmp_file_path)

                    # Check if extraction worked
                    if os.path.exists(dest_file):
                        extracted_files.append(dest_file)
                        filename = match.split('/')[-1].replace('.pickle', '')
                        print(f"Extracted from {source_type}: {filename}")
                        continue  # Success

                except Exception as cmd_error:
                    print(f"   Tar command method failed: {cmd_error}")
                    # Clean up temp file if it exists
                    try:
                        if 'tmp_file_path' in locals():
                            os.unlink(tmp_file_path)
                    except:
                        pass

                # Method 3: Try with individual tar extraction using wildcards
                try:
                    print(f"   Trying individual tar extraction...")

                    # Use tar to list and extract, being very explicit
                    list_cmd = ['tar', '-tzf', tar_file]
                    result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)

                    # Find the exact match in the list
                    all_files = result.stdout.strip().split('\n')
                    exact_match = None
                    for file_in_tar in all_files:
                        if file_in_tar == match:
                            exact_match = file_in_tar
                            break

                    if exact_match:
                        # Extract using Python's tarfile with the exact member
                        with tarfile.open(tar_file, 'r:gz') as tar:
                            tar.extract(exact_match, path=output_dir if output_dir else '.')

                        if os.path.exists(dest_file):
                            extracted_files.append(dest_file)
                            filename = match.split('/')[-1].replace('.pickle', '')
                            print(f"Extracted from {source_type}: {filename}")
                        else:
                            print(f"   File not found after extraction: {dest_file}")
                    else:
                        print(f"   Could not find exact match in tar file list")

                except Exception as individual_error:
                    print(f"   Individual extraction failed: {individual_error}")

            except Exception as e:
                print(f"Unexpected error extracting {match}: {e}")

        return extracted_files

    except subprocess.CalledProcessError as e:
        print(f"Error listing tar contents: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def copy_pickle_to_folder(extracted_files, destination_folder):
    """
    Copy extracted pickle files to a specific folder with simplified names.

    Args:
        extracted_files (list): List of extracted file paths
        destination_folder (str): Folder to copy files to

    Returns:
        list: List of copied file paths
    """
    if not extracted_files:
        print("No files to copy")
        return []

    # Create destination folder
    os.makedirs(destination_folder, exist_ok=True)
    print(f"\nCopying files to: {destination_folder}")

    copied_files = []

    for file_path in extracted_files:
        if os.path.exists(file_path):
            try:
                # Get the original filename
                original_filename = os.path.basename(file_path)

                # Create destination path
                dest_path = os.path.join(destination_folder, original_filename)

                # Copy the file
                import shutil
                shutil.copy2(file_path, dest_path)

                copied_files.append(dest_path)
                print(f"Copied: {original_filename}")

            except Exception as e:
                print(f"Failed to copy {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    return copied_files


def main():
    """
    Main function to extract specific SMILES from tar archive.
    Edit the variables below to set your input parameters.
    """

    # ========== INPUT PARAMETERS - EDIT THESE ==========
    tar_file = "rdkit_folder.tar.gz"  # Path to your tar file
    target_smiles = "CSC[C@H](N)C(=O)O"  # Your SMILES (always uppercase)
    output_dir = "./extracted_temp"  # Temporary extraction directory
    final_folder = "./GEOM_molecules"  # Final folder to save pickle files
    # ===================================================

    print("=" * 60)
    print("SMILES EXTRACTION FROM TAR ARCHIVE")
    print("QM9 + DRUGS DATASETS")
    print("=" * 60)
    print(f"Input SMILES (uppercase): {target_smiles}")
    print(f"Tar file: {tar_file}")
    print(f"Temporary extraction: {output_dir}")
    print(f"Final molecule folder: {final_folder}")
    print(f"Searching in: rdkit_folder/qm9/ and rdkit_folder/drugs/")

    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Temporary directory created: {output_dir}")

    # Extract the specific SMILES
    extracted_files = extract_specific_smiles(
        tar_file=tar_file,
        target_smiles=target_smiles,
        output_dir=output_dir
    )

    # Copy files to final folder
    if extracted_files:
        copied_files = copy_pickle_to_folder(extracted_files, final_folder)

        # Clean up temporary extraction directory (optional)
        import shutil
        try:
            shutil.rmtree(output_dir)
            print(f"Cleaned up temporary directory: {output_dir}")
        except:
            print(f"WARNING: Could not clean up temporary directory: {output_dir}")

        # Final summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)

        if copied_files:
            print(f"Successfully saved {len(copied_files)} molecule file(s) to {final_folder}:")
            for file_path in copied_files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    filename = os.path.basename(file_path)
                    # Determine source dataset
                    source = "[Unknown]"
                    for ef in extracted_files:
                        if filename in ef:
                            if 'rdkit_folder/qm9/' in ef:
                                source = "[QM9]"
                            elif 'rdkit_folder/drugs/' in ef:
                                source = "[Drugs]"
                            break
                    print(f"   {source} {filename} ({file_size:,} bytes)")

            print(f"\nYour molecule pickle file(s) are ready in: {final_folder}")
        else:
            print("ERROR: No files were saved to final folder")

    else:
        print(f"\nERROR: No matches found for SMILES: {target_smiles}")



if __name__ == "__main__":
    # Run the main function with hardcoded values
    main()