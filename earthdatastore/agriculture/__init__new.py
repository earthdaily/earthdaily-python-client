from earthdatastore._api_requester import APIRequester
from earthdatastore.agriculture._migrated_from_v0 import parallel_search


class AgricultureService:
    """
    Represents the Agriculture Service for interacting with specific agriculture-related methods.

    Attributes:
    -----------
    api_requester : APIRequester
        An instance of APIRequester used to send HTTP requests to the EDS API.
    """

    def __init__(self, api_requester: APIRequester):
        """
        Initialize the AgricultureService.

        Parameters:
        -----------
        api_requester : APIRequester
            An instance of APIRequester used to send HTTP requests to the EDS API.
        """
        self.api_requester = api_requester
        self.parallel_search = parallel_search
