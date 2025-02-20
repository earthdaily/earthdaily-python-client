import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from earthdaily._zipper import Zipper


class TestZipper(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp())
        self.zipper = Zipper()

        # Create test files and directories
        (self.test_dir / "test_file.txt").write_text("Test content")
        test_subdir = self.test_dir / "subdir"
        test_subdir.mkdir()
        (test_subdir / "subdir_file.txt").write_text("Subdir test content")

    def tearDown(self):
        # Clean up temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_zip_single_file(self):
        source_file = self.test_dir / "test_file.txt"
        output_path = self.test_dir / "output_single.zip"

        result_path = self.zipper.zip_content(source_file, output_path)

        self.assertTrue(result_path.exists())
        self.assertTrue(zipfile.is_zipfile(result_path))

        with zipfile.ZipFile(result_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            self.assertEqual(len(file_list), 1)
            self.assertEqual(file_list[0], "test_file.txt")

    def test_zip_directory(self):
        output_path = self.test_dir / "output_dir.zip"

        result_path = self.zipper.zip_content(self.test_dir, output_path)

        self.assertTrue(result_path.exists())
        self.assertTrue(zipfile.is_zipfile(result_path))

        with zipfile.ZipFile(result_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            self.assertEqual(len(file_list), 3)
            self.assertIn("test_file.txt", file_list)
            self.assertIn("subdir/subdir_file.txt", file_list)

    def test_nonexistent_source(self):
        nonexistent_path = self.test_dir / "nonexistent"
        output_path = self.test_dir / "output_nonexistent.zip"

        with self.assertRaises(FileNotFoundError):
            self.zipper.zip_content(nonexistent_path, output_path)

    def test_output_path_creation(self):
        source_file = self.test_dir / "test_file.txt"
        nested_dir = self.test_dir / "nested"
        nested_dir.mkdir(exist_ok=True, parents=True)  # Create the nested directory
        output_path = nested_dir / "output.zip"

        result_path = self.zipper.zip_content(source_file, output_path)

        self.assertTrue(result_path.exists())
        self.assertTrue(zipfile.is_zipfile(result_path))

    def test_zip_content_integrity(self):
        source_file = self.test_dir / "test_file.txt"
        output_path = self.test_dir / "output_integrity.zip"

        self.zipper.zip_content(source_file, output_path)

        with tempfile.TemporaryDirectory() as extract_dir:
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            extracted_file = Path(extract_dir) / "test_file.txt"
            self.assertTrue(extracted_file.exists())
            self.assertEqual(extracted_file.read_text(), "Test content")


if __name__ == "__main__":
    unittest.main()
