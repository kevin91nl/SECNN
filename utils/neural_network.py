class DummyLoss:
    """A dummy loss class such that it has a backward method (which does nothing).
    """

    def backward(self, *args, **kwargs):
        """A backward method (which does nothing).

        Parameters
        ----------
        args : list
            Arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        Nothing.
        """
        pass
