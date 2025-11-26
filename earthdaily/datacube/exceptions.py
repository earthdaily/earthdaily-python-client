class DatacubeError(Exception):
    pass


class DatacubeCreationError(DatacubeError):
    pass


class DatacubeMaskingError(DatacubeError):
    pass


class DatacubeMergeError(DatacubeError):
    pass


class DatacubeOperationError(DatacubeError):
    pass


class DatacubeValidationError(DatacubeError):
    pass


class DatacubeVisualizationError(DatacubeError):
    pass
