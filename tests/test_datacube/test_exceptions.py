import unittest

from earthdaily.datacube.exceptions import (
    DatacubeCreationError,
    DatacubeError,
    DatacubeMaskingError,
    DatacubeMergeError,
    DatacubeOperationError,
    DatacubeValidationError,
    DatacubeVisualizationError,
)


class TestExceptions(unittest.TestCase):
    def test_datacube_error_is_exception(self):
        self.assertTrue(issubclass(DatacubeError, Exception))

    def test_datacube_creation_error_inherits_from_datacube_error(self):
        self.assertTrue(issubclass(DatacubeCreationError, DatacubeError))

    def test_datacube_creation_error_can_be_raised(self):
        with self.assertRaises(DatacubeCreationError):
            raise DatacubeCreationError("Test error")

    def test_datacube_creation_error_message(self):
        error_msg = "Test creation error"
        with self.assertRaises(DatacubeCreationError) as context:
            raise DatacubeCreationError(error_msg)
        self.assertEqual(str(context.exception), error_msg)

    def test_datacube_masking_error_inherits_from_datacube_error(self):
        self.assertTrue(issubclass(DatacubeMaskingError, DatacubeError))

    def test_datacube_masking_error_can_be_raised(self):
        with self.assertRaises(DatacubeMaskingError):
            raise DatacubeMaskingError("Test masking error")

    def test_datacube_masking_error_message(self):
        error_msg = "Test masking error"
        with self.assertRaises(DatacubeMaskingError) as context:
            raise DatacubeMaskingError(error_msg)
        self.assertEqual(str(context.exception), error_msg)

    def test_datacube_merge_error_inherits_from_datacube_error(self):
        self.assertTrue(issubclass(DatacubeMergeError, DatacubeError))

    def test_datacube_merge_error_can_be_raised(self):
        with self.assertRaises(DatacubeMergeError):
            raise DatacubeMergeError("Test merge error")

    def test_datacube_merge_error_message(self):
        error_msg = "Test merge error"
        with self.assertRaises(DatacubeMergeError) as context:
            raise DatacubeMergeError(error_msg)
        self.assertEqual(str(context.exception), error_msg)

    def test_datacube_operation_error_inherits_from_datacube_error(self):
        self.assertTrue(issubclass(DatacubeOperationError, DatacubeError))

    def test_datacube_operation_error_can_be_raised(self):
        with self.assertRaises(DatacubeOperationError):
            raise DatacubeOperationError("Test operation error")

    def test_datacube_operation_error_message(self):
        error_msg = "Test operation error"
        with self.assertRaises(DatacubeOperationError) as context:
            raise DatacubeOperationError(error_msg)
        self.assertEqual(str(context.exception), error_msg)

    def test_datacube_validation_error_inherits_from_datacube_error(self):
        self.assertTrue(issubclass(DatacubeValidationError, DatacubeError))

    def test_datacube_validation_error_can_be_raised(self):
        with self.assertRaises(DatacubeValidationError):
            raise DatacubeValidationError("Test validation error")

    def test_datacube_validation_error_message(self):
        error_msg = "Test validation error"
        with self.assertRaises(DatacubeValidationError) as context:
            raise DatacubeValidationError(error_msg)
        self.assertEqual(str(context.exception), error_msg)

    def test_datacube_visualization_error_inherits_from_datacube_error(self):
        self.assertTrue(issubclass(DatacubeVisualizationError, DatacubeError))

    def test_datacube_visualization_error_can_be_raised(self):
        with self.assertRaises(DatacubeVisualizationError):
            raise DatacubeVisualizationError("Test visualization error")

    def test_datacube_visualization_error_message(self):
        error_msg = "Test visualization error"
        with self.assertRaises(DatacubeVisualizationError) as context:
            raise DatacubeVisualizationError(error_msg)
        self.assertEqual(str(context.exception), error_msg)


if __name__ == "__main__":
    unittest.main()
