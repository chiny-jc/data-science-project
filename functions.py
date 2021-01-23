''' -------------------------------------- FUNCTIONS -------------------------------------- '''

import re 
def add_unique_elements(list_to_iterate, set_to_add):
    for element in list_to_iterate:
        set_to_add.add(element)
        
def regex_lookup(column, regex_pattern, match_only=True):
    matches = []
    idx = []
    for i in range(len(column) - 1):
        match = re.search(regex_pattern, column.iloc[i])
        if match != None:
            matches.append(match.group(0))
            idx.append(i)
    
    if match_only:
        return matches
    else:
        return column.iloc[idx]
    
def calculate_distance_between_coordinates(first_coordinates, second_coordinates):
    meters_per_coordinate_degree = 111.139
    latitude_difference_in_km = abs(first_coordinates[0] - second_coordinates[0]) * meters_per_coordinate_degree
    longitude_difference_in_km = abs(first_coordinates[1] - second_coordinates[1]) * meters_per_coordinate_degree
    linear_distance = np.sqrt(latitude_difference_in_km ** 2 + longitude_difference_in_km ** 2)
    return linear_distance


def text_cleaner(text):
    text = text.replace(".", "")
    text = text.replace("'", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", " ")
    text = text.replace("=", "")
    text = text.replace("1", "")
    text = text.replace("2", "")
    text = text.replace("3", "")
    text = text.replace("4", "")
    text = text.replace("5", "")
    text = text.replace("6", "")
    text = text.replace("7", "")
    text = text.replace("8", "")
    text = text.replace("9", "")
    text = text.replace("0", "")
    text = text.replace("%", "")
    text = text.replace("<", "")
    text = text.replace(">", "")
    text = text.replace("&", "")
    text = text.replace("§", "")
    text = text.replace("/", "")
    text = text.replace("+", "")
    text = text.replace("*", "")
    text = text.replace("#", "")
    text = text.replace("br", "")
    

    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()

