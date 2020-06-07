from Features import Features
class Concept:
    title, url, content, id, domain = [None, None, None, None, None]

    def __init__(self, id, url, title, content, domain):
        self.id = id
        self.url = url
        self.title = title
        self.content = content
        self.domain = domain
        self.features = Features()

    def __eq__(self, other):
        return self.title == other  # Needed for 'IndexOf' when paring the desired Graph Matrix