class Model:
    """Root object Containing all parsed/elaborated info

    ...

    Attributes
    ----------
    dataset : [Concept]
        Contains all parsed Concepts, each Concept subsequently contains its own features
    desiredGraph : GraphMatrix
        Contains the desired parsed labels on which to train the NN

    """
    dataset = []
    pairFeatures = None
    desiredGraph = None
