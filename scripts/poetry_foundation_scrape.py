from bs4 import BeautifulSoup
from requests_html import HTMLSession
from tqdm import tqdm

import json
import requests
import unicodedata


def get_poem(poem_url):
    poem = []
    page = requests.get(poem_url)
    soup = BeautifulSoup(page.content, "html.parser")

    #Find poem lines
    poem_list_component = soup.find_all("div", attrs={'style':'text-indent: -1em; padding-left: 1em;'})

    title_temp = soup.find("h1", attrs={'class':'c-hdgSans c-hdgSans_2 c-mix-hdgSans_inline'})
    title = " ".join(title_temp.text.split()) if title_temp else "NO_TITLE"
    
    #Author
    results = soup.find("span", attrs={'class':'c-txt c-txt_attribution'})
    if results:
        author_temp = results.find("a")
        author = author_temp.text if author_temp else "NO_AUTHOR"
    else:
        author = "NO_AUTHOR"

    #Grab lines in unicode format.
    for i in range(len(poem_list_component)):
        p = unicodedata.normalize("NFKD", poem_list_component[i].text)
        if i < len(poem_list_component) - 1:
            poem.append(p + "\n")
        else:
            poem.append(p)
    
    return title, author, poem

def get_split_poem(poem, dl=8):
    """
        Generate poems from one poem by trimming and trying to keep
        as much of the structural elements and cohesive structure.
    """
    poems = []

    #Generate intervals
    upper_itv = []
    s = 0
    for i in range(1, len(poem)):
        if len(poem[i]) > 0:
            if poem[i][0].isupper():
                upper_itv.append(i - s)
                s=i
    upper_itv.append(len(poem) - s)

    size = 0
    j = 0
    for i in range(len(upper_itv)):
        c_size = upper_itv[i]
        if size + c_size < dl:
            if size + c_size < 3 and i == len(upper_itv) - 1:
                if len(poems) == 0:
                    poems.append("".join(poem[j:j+c_size]))
                elif size == 0:
                    poems[-1] += "".join(poem[j:j+c_size])
                else:
                    poems[-2] += poems[-1] + "".join(poem[j:j+c_size])
                    del poems[-1]
            else:
                size += c_size
                poems.append("".join(poem[j:j+c_size]))
                j += c_size
        else:
            extra_l = (upper_itv[i] + size) % dl
            if extra_l < 3:
                if upper_itv[i] + size > dl:
                    parts = int((upper_itv[i] + size - extra_l) / dl)
                else:
                    parts = 1
                for k in range(parts):
                    if k == 0:
                        if size > 0:
                            poems[-1] += "".join(poem[j:j + (dl-size) + extra_l])
                        else:
                            poems.append("".join(poem[j:j + (dl-size) + extra_l]))
                        j += (dl-size) + extra_l
                        size = 0
                    else:
                        poems.append("".join(poem[j: j+dl]))
                        j += dl
            else:
                if upper_itv[i] + size > dl:
                    parts = int(((upper_itv[i] + size - extra_l) / dl) + 1)
                else:
                    parts = 1
                for k in range(parts):
                    if k == 0:
                        if size > 0:
                            poems[-1] += "".join(poem[j:j+extra_l])
                        else:
                            poems.append("".join(poem[j:j+extra_l]))

                        j += extra_l
                        size = 0

                    else:
                        poems.append("".join(poem[j:j+dl]))
                        j+=dl
    return poems

def main():
    MAX_PAGE=243
    FLAG = False
    ID = 1
    not_processed = []
    data = {
        "info" : "Free-verse poems extracted from Poetry Foundation.org and split into small sub-poems of length 8ish",
        "poems" : []
    }

    for PAGE in tqdm(range(23, MAX_PAGE+1)):
        session = HTMLSession()

        MAIN_URL="""https://www.poetryfoundation.org/poems/browse#page={page}&sort_by=recently_added&forms=259"""

        r = session.get(MAIN_URL.format(page=PAGE))

        r.html.render(sleep=3, keep_page=True, scrolldown=3, timeout=20)
        poem_links = r.html.find('a')
        num_poems = 0

        #Variables stored until we have all themes associated 
        #with the poem and can output to our data object
        title = ""
        author = ""
        poems = []
        tags = []

        for a_href in poem_links:
            if 'class' in a_href.attrs and a_href.attrs['class'] == ('c-pagination-control',):
                if FLAG:
                    if num_poems == 0: not_processed.append(PAGE)
                    FLAG = False

            if FLAG:
                #Poem -- next tags will be themes associated with poem
                if len(a_href.attrs) == 1:
                    #Output JSON we have encountered a new poem so we
                    #have iterated through all of the tags.
                    if len(poems) > 0:
                        for p in poems:
                            if p != "":
                                data['poems'].append({
                                    "id" : ID,
                                    "poem" : p,
                                    "title" : title,
                                    "author" : author,
                                    "keywords" : tags
                                })

                                ID += 1

                    title, author, poem = get_poem(a_href.attrs['href'])
                    poems = get_split_poem(poem, 8)
                    tags = []
                else:
                    tag = a_href.text
                    if tag and tag != "Appeared in Poetry Magazine":
                        tags.append(tag)

            if ('href' in a_href.attrs and a_href.attrs['href'] == '#') and \
                ('class' in a_href.attrs and a_href.attrs['class'] == ()):
                FLAG = True

        with open(f'data/poem/{PAGE}.json', 'w') as f:
            json.dump(data, f)
        session.close()

    return data, not_processed

if __name__ == '__main__':
    data, not_processed = main()

    with open('daddy.json', 'w') as f:
        json.dump(data, f)

    print(not_processed)

    # title, author, poem = get_poem("https://www.poetryfoundation.org/poetrymagazine/poems/146507/bible-belt")
    # print(title)
    # print(author)
    # poems = get_split_poem(poem, 8)
    # for i in poems:
    #     print(len(i.split("\n")))
    #     print(i)