import requests

PAGE=1
MAIN_URL="""https://www.poetryfoundation.org/poems/browse#page={page}&sort_by=recently_added&forms=259"""

def update_page():
    """
        Updates page number and returns the new main page URL
        :return: String
    """
    PAGE+=1
    return MAIN_URL.format(page=PAGE)

