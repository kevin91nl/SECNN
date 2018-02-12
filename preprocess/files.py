import json


class JSONFileLoader:
    """The file loader class used for a data iterator such that it loads JSON data when requested."""

    def __init__(self, preprocessor=None):
        """Initialize the file loader.

        Parameters
        ----------
        preprocessor : method, optional
            Method which is applied to the data when loaded (default: None).
        """
        self.preprocessor = preprocessor

    def load_file(self, path):
        """Loads (open, read and JSON decode) a file.

        Parameters
        ----------
        path : str
            The path of the file.

        Returns
        -------
        dict
            The preprocessed version of the data found in the file.
        """
        with open(path, 'r') as input_handle:
            data = json.load(input_handle)
        if self.preprocessor is None:
            return data
        else:
            return self.preprocessor(data)
