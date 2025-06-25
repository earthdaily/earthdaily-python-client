import os
import zipfile
from pathlib import Path


class Zipper:
    def __init__(self):
        """
        Initializes a new instance of the Zipper class.
        """
        self.compression = zipfile.ZIP_STORED

    def zip_content(self, source: Path, output_path: Path) -> Path:
        """
        Zips the contents specified by source path and saves it to a specified output path.

        Parameters:
        source (Path): The path to the directory or file to be zipped.
        output_path (Path): The path (including filename) where the zip file will be saved.

        Returns:
        Path: The path to the created zip file.
        """
        if not source.exists():
            raise FileNotFoundError("The provided source path does not exist.")

        with zipfile.ZipFile(output_path, "w", self.compression) as zipf:
            if source.is_dir():
                for root, dirs, files in os.walk(source):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(source)
                        zipf.write(file_path, arcname)
            else:
                zipf.write(source, source.name)
        return output_path
