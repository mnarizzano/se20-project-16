class Meta:
    """ Engine

    Container for all single Concept Meta Features

    ...

    Attributes
    ----------
    categories : [string]
        List of Wikipedia categories the Concept belongs to
    links : [string]
        List of Wikipedia pages the Concept points to

    """
    scategories = []
    links = []

    def get_categories(self):
        return self.categories

    def get_links(self):
        return self.links
