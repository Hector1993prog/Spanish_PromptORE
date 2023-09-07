from collections import Counter
import re

def entity_sort_dictionary(result_list):
    """
    Sorts phrases in a list based on the count of entities and returns a dictionary.

    Args:
        result_list (list): A list of phrases.

    Returns:
        dict: A dictionary where keys represent entity counts and values are lists of sorted phrases.
    """
    phrases_by_entity_count = {}

    # Iterate through the result_list
    for phrase in result_list:
        # Count the number of entities in the phrase
        entity_count = len(re.findall(r'\$[^$]+\$', phrase))

        # Add the phrase to the appropriate list in the dictionary
        if entity_count not in phrases_by_entity_count:
            phrases_by_entity_count[entity_count] = []

        # Add the period '.' to the end of each phrase
        phrases_by_entity_count[entity_count].append(phrase.strip() + '.')

    # Sort the keys and create a new dictionary with sorted keys
    sorted_keys = sorted(phrases_by_entity_count.keys())
    sorted_phrases_by_entity_count = {key: phrases_by_entity_count[key] for key in sorted_keys}

    return sorted_phrases_by_entity_count


def counter(dict_list):
    """
    Counts the number of values in each list within a dictionary and returns a Counter.

    Args:
        dict_list (dict): A dictionary where keys represent a category and values are lists.

    Returns:
        Counter: A Counter object with category counts.
    """
    result = Counter()

    for k, values in dict_list.items():
        result[k] = len(values)

    return result
