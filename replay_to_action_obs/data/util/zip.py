import os
import zipfile

def zip_dataset(directory_path: str) -> None:
    """
    Zips the contents of a directory and stores the zip file within the same directory.

    Args:
        directory_path (str): The path of the directory to zip.
        zip_filename (str): The name of the zip file to be created.
    """
    # Full path for the zip file to be stored in the directory
    dir_name = os.path.basename(os.path.normpath(directory_path))
    zip_path = os.path.join(directory_path, f"{dir_name}.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk the directory
        for foldername, subfolders, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                
                # Skip the zip file itself
                if file_path == zip_path:
                    continue
                
                # Add file to the zip file
                arcname = os.path.relpath(file_path, directory_path)
                zip_file.write(file_path, arcname)