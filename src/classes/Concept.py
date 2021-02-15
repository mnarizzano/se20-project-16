from Features import Features
from Meta import Meta

class Concept:
    """ Concept

    This class is the basic container for a single concept
    It stores both basic properties from the input file as url,
    conceptId and text and complex ones extracted during elaboration
    such as metadata and raw featurew
    """

    title, url, content, id, domain = [None, None, None, None, None]

    def __init__(self, id, url, title, content, domain):
        """Returns a Concept object initialized with informations from the input
           file and empty meta/raw data

        """
        self.id = id
        self.url = url
        self.title = title
        self.content = content
        self.domain = domain
        self.features = Features()
        self.meta = Meta()

    def __eq__(self, other):
        """Implements the equals to operator for this custom class

        Parameters
        ----------
        other : Concept
            object to compare to

        Returns
        -------
        bool
        """
        return self.title == other  # Needed for 'IndexOf' when paring the desired Graph Matrix

    def getFeatures(self):
        """Getter for Raw Features of this concept

        Returns
        -------
        Features
        """
        return self.features