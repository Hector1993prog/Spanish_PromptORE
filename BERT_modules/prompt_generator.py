import regex as re
import json
import lxml.etree as ET
from auxiliary_function_XML_process import entity_sort_dictionary, counter


def phrase_extraction(
    xml_path: str,
    return_dict: bool = True,
    ns: str = None,
    parragraph_objetives: str = None,
    entity_objetives: str = None,
    count: bool = False
) -> dict:
    '''
    DESCRIPTION: 
    This function extracts the full text of a XML-TEI file 
    and divides it into paragraphs. After it divides the paragraphs
    into a list of phrases based on the number of entities located in those
    phrases.

    INPUTS:
        xml_path: The XML directory path.

        ns: 
        name space dictionary: default: {'tei': 'http://www.tei-c.org/ns/1.0'}

        parragraph_objetives: 
        the paragraph tags to look for. 
        default: 
        ['{http://www.tei-c.org/ns/1.0}pb', '{http://www.tei-c.org/ns/1.0}p']

        entity_objetives: 
        The tag of the entities to mark.
        default:
        ["{http://www.tei-c.org/ns/1.0}persName",
        "{http://www.tei-c.org/ns/1.0}orgName",
        "{http://www.tei-c.org/ns/1.0}placeName",
        "{http://www.tei-c.org/ns/1.0}date",
        "{http://www.tei-c.org/ns/1.0}rs",
        "{http://www.tei-c.org/ns/1.0}fw"  # Include fw element
        ]

        count: 
        optional parameter that returns a numerical dictionary for the keys.
        default: False

    OUTPUTS: dictionary with keys for the number of entities and the list of phrases attached.
    '''

    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    parragraph_objetives = ['{http://www.tei-c.org/ns/1.0}pb', '{http://www.tei-c.org/ns/1.0}p']

    entity_objetives = [
        "{http://www.tei-c.org/ns/1.0}persName",
        "{http://www.tei-c.org/ns/1.0}orgName",
        "{http://www.tei-c.org/ns/1.0}placeName",
        "{http://www.tei-c.org/ns/1.0}date",
        "{http://www.tei-c.org/ns/1.0}rs",
        "{http://www.tei-c.org/ns/1.0}fw"  # Include fw element
    ]

    result = ''

    for div in root.xpath('//tei:div', namespaces=ns):
        for element in div:
            if element.tag in parragraph_objetives and element.text is not None:
                # Include all text content within the <p> element
                paragraph_text = element.text
                for entity in element:
                    if entity.tag in entity_objetives and entity.text is not None:
                        # Include entity tags as well
                        paragraph_text += ' $' + re.sub(r'\s+', ' ', entity.text) + '$ '
                    if entity.tail is not None:
                        # Include tail text of entities
                        paragraph_text += entity.tail
                paragraph_text += (element.tail or '')

                result += paragraph_text  # Add the entire text

    result = ' '.join(result.split())
    result = re.sub(r'\s+\-', '', result)
    result = re.sub(r'[\[\]\(\)]', '', result)
    result = re.sub(r'\:.', ': ', result)
    result = re.sub(r'\ , ', ', ', result)
    result = re.sub(r'\* * * ', '', result)

    result_list = re.sub(r'\[\S+\]', '', result).split('.')

    if count and return_dict:
        return entity_sort_dictionary(result_list), counter(entity_sort_dictionary(result_list))
    elif not count and return_dict:
        return entity_sort_dictionary(result_list)
    elif count and not return_dict:
        result_list = [i + '.' if not i.endswith(':') else i for i in result_list]
        return counter(entity_sort_dictionary(result_list)), result_list
    else:
        result_list = [i + '.' if not i.endswith(':') else i for i in result_list]
        return result_list




def prompt_generator(
        
        phrase_input: dict = None,
        full_extraction: bool = False,
        entity_number: int = 2,
        json_file_path_name: str = 'prompts_beto.json',
        
        )-> dict:
    '''
    DESCRIPTION:

    This function takes a dictionary which has the number of entities as keys
    and returns a dictionary of phrases with a PromptORE structure attached.
    
    Args::

    phrase_input: The input dictionary containing phrases for each entity number.
    
    full_extraction: A boolean flag indicating whether full extraction should be done.

    entity_number: The number of entities key objective for the prompting. Now it supports 1, 2, and 3 entities.
    
    json_file_path_name: The name of the path to save the dictionary in a JSON format.
    
    Returns: 
    
    prompt_dict: A dictionary with the prompt structure as keys and the list of phrases with the prompts as values.
    '''
    if phrase_input is None:
        raise ValueError('The output dictionary from phrase_extraction function is required')
    
    entity_dictionary = phrase_input
    prompt_dict = {}

    if full_extraction:
        for k, v in entity_dictionary.items():

            if k == 2:
                phrases = entity_dictionary.get(k, v)

                prompt_0_list_2_ent = []
                prompt_1_list_2_ent = []
                prompt_2_list_2_ent = []
                prompt_3_list_2_ent = []
                prompt_4_list_2_ent = []
                prompt_5_list_2_ent = []
                for phrase in phrases:

                    entities = re.findall(r'\$(.*?)\$', phrase)
                    entity_names = [entity.strip() for entity in entities]
                    cleaned_phrase = re.sub(r'\$', '', phrase)

                    prompt_0 = f'[CLS] {cleaned_phrase} {entity_names[0]} [MASK] {entity_names[1]}. [SEP]'
                    prompt_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es una relación de tipo [MASK]. [SEP]'
                    prompt_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es una relación de [MASK]. [SEP]'   
                    prompt_3 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es de naturaleza [MASK] [SEP]'
                    prompt_4 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es de carácter [MASK] [SEP]'
                    prompt_5_1 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[0]} y {entity_names[1]}? La relación es el [MASK] [SEP]'
                    prompt_5_2 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[0]} y {entity_names[1]}? La relación es la [MASK] [SEP]'    
   
                    prompt_0_list_2_ent.append(prompt_0)
                    prompt_1_list_2_ent.append(prompt_1)
                    prompt_2_list_2_ent.append(prompt_2)
                    prompt_0_list_2_ent.append(prompt_0)
                    prompt_1_list_2_ent.append(prompt_1)
                    prompt_2_list_2_ent.append(prompt_2)
                    prompt_3_list_2_ent.append(prompt_3)
                    prompt_4_list_2_ent.append(prompt_4)
                    prompt_5_list_2_ent.append(prompt_5_1)
                    prompt_5_list_2_ent.append(prompt_5_2)
                prompt_dict['prompt_0_ent_2'] = prompt_0_list_2_ent
                prompt_dict['prompt_1_ent_2'] = prompt_1_list_2_ent 
                prompt_dict['prompt_2_ent_2'] = prompt_2_list_2_ent
                prompt_dict['prompt_3_ent_2'] = prompt_3_list_2_ent
                prompt_dict['prompt_4_ent_2'] = prompt_4_list_2_ent
                prompt_dict['prompt_5_ent_2'] = prompt_5_list_2_ent

            elif k == 3:

                phrases = entity_dictionary.get(k, v)

                prompt_0_list_3_ent = []
                prompt_1_list_3_ent  = []
                prompt_2_list_3_ent = []
                prompt_3_list_3_ent = []
                prompt_4_list_3_ent = []
                prompt_5_list_3_ent = []   
   

                for phrase in phrases:

                    entities = re.findall(r'\$(.*?)\$', phrase)
                    entity_names = [entity.strip() for entity in entities]
                    cleaned_phrase = re.sub(r'\$', '', phrase)

                    prompt_0_0 = f'[CLS] {cleaned_phrase} {entity_names[0]} [MASK] {entity_names[1]}. [SEP]'
                    prompt_0_1 = f'[CLS] {cleaned_phrase} {entity_names[1]} [MASK] {entity_names[2]}. [SEP]'
                    prompt_0_2 = f'[CLS] {cleaned_phrase} {entity_names[0]} [MASK] {entity_names[2]}. [SEP]'
                        
                    prompt_1_0 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es una relación de tipo [MASK]. [SEP]'
                    prompt_1_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[1]} y {entity_names[2]} es una relación de tipo [MASK]. [SEP]'
                    prompt_1_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[2]} y {entity_names[0]} es una relación de tipo [MASK]. [SEP]'

                    prompt_2_0 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es una relación de [MASK]. [SEP]'
                    prompt_2_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[1]} y {entity_names[2]} es una relación de [MASK]. [SEP]'
                    prompt_2_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[2]} y {entity_names[0]} es una relación de [MASK]. [SEP]'
                    
                    prompt_3_0 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es de naturaleza [MASK]. [SEP]'
                    prompt_3_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[1]} y {entity_names[2]} es de naturaleza [MASK]. [SEP]'
                    prompt_3_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[2]} y {entity_names[0]} es de naturaleza [MASK]. [SEP]'

                    prompt_4_0 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es de carácter [MASK]. [SEP]'
                    prompt_4_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[1]} y {entity_names[2]} es de carácter [MASK]. [SEP]'
                    prompt_4_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[2]} y {entity_names[0]} es de carácter [MASK]. [SEP]'

                    prompt_5_1_0 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[0]} y {entity_names[1]}? La relación es el [MASK]. [SEP]'
                    prompt_5_1_1 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[1]} y {entity_names[2]}? La relación es el [MASK]. [SEP]'
                    prompt_5_1_2 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[2]} y {entity_names[0]}? La relación es el [MASK]. [SEP]'
                    prompt_5_2_0 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[0]} y {entity_names[1]}? La relación es la [MASK]. [SEP]'
                    prompt_5_2_1 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[1]} y {entity_names[2]}? La relación es la [MASK]. [SEP]'
                    prompt_5_2_2 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[2]} y {entity_names[0]}? La relación es la [MASK]. [SEP]'
                    
                    prompt_0_list_3_ent.append(prompt_0_0)
                    prompt_0_list_3_ent.append(prompt_0_1)
                    prompt_0_list_3_ent.append(prompt_0_2)

                    prompt_1_list_3_ent.append(prompt_1_0)
                    prompt_1_list_3_ent.append(prompt_1_1)
                    prompt_1_list_3_ent.append(prompt_1_2)

                    prompt_2_list_3_ent.append(prompt_2_0)
                    prompt_2_list_3_ent.append(prompt_2_1)  
                    prompt_2_list_3_ent.append(prompt_2_2) 

                    prompt_3_list_3_ent.append(prompt_3_0)
                    prompt_3_list_3_ent.append(prompt_3_1)  
                    prompt_3_list_3_ent.append(prompt_3_2) 

                    prompt_4_list_3_ent.append(prompt_4_0)
                    prompt_4_list_3_ent.append(prompt_4_1)  
                    prompt_4_list_3_ent.append(prompt_4_2) 

                    prompt_5_list_3_ent.append(prompt_5_1_0)
                    prompt_5_list_3_ent.append(prompt_5_1_1)  
                    prompt_5_list_3_ent.append(prompt_5_1_2) 
                    prompt_5_list_3_ent.append(prompt_5_2_0)
                    prompt_5_list_3_ent.append(prompt_5_2_1)  
                    prompt_5_list_3_ent.append(prompt_5_2_2) 

                prompt_dict['prompt_0_ent_3'] = prompt_0_list_3_ent
                prompt_dict['prompt_1_ent_3'] = prompt_1_list_3_ent
                prompt_dict['prompt_2_ent_3'] = prompt_2_list_3_ent
                prompt_dict['prompt_3_ent_3'] = prompt_3_list_3_ent
                prompt_dict['prompt_4_ent_3'] = prompt_4_list_3_ent
                prompt_dict['prompt_5_ent_3'] = prompt_5_list_3_ent

            elif k == 1:

                phrases = entity_dictionary.get(k, v)
                    
                prompt_list_unique = []
                prompt_list_unique_1 = []

                for phrase in phrases:

                    entities = re.findall(r'\$(.*?)\$', phrase)
                    entity_names = [str(entity.strip()) for entity in entities]
                    cleaned_phrase = re.sub(r'\$', '', phrase)
                        

                        
                    prompt_0_unique = f"[CLS] {cleaned_phrase} La relación entre {''.join(entity_names)} y la frase anterior es una relación de [MASK]. [SEP]"
                    prompt_1_unique= f"[CLS] {cleaned_phrase} La relación entre {''.join(entity_names)} y la frase anterior es una relación de tipo [MASK]. [SEP]"

                    prompt_list_unique.append(prompt_0_unique)        
                    prompt_list_unique_1.append(prompt_1_unique)

                prompt_dict['prompt_0_unique'] = prompt_list_unique
                prompt_dict['prompt_1_unique'] = prompt_list_unique_1

        if json_file_path_name:

            with open(json_file_path_name, 'w', encoding='utf-8') as json_file:

                json.dump(prompt_dict, json_file)

        return prompt_dict
    
    else:
          # If full_extraction is False
        if entity_number in entity_dictionary and entity_number == 2:

            phrases = entity_dictionary.get(entity_number, [])

            prompt_0_list_2_ent = []
            prompt_1_list_2_ent = []
            prompt_2_list_2_ent = []
            prompt_3_list_2_ent = []
            prompt_4_list_2_ent = []
            prompt_5_list_2_ent = []

            for phrase in phrases:

                entities = re.findall(r'\$(.*?)\$', phrase)
                entity_names = [entity.strip() for entity in entities]
                cleaned_phrase = re.sub(r'\$', '', phrase)

                prompt_0 = f'[CLS] {cleaned_phrase} {entity_names[0]} [MASK] {entity_names[1]}. [SEP]'
                prompt_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es una relación de tipo [MASK]. [SEP]'
                prompt_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es una relación de [MASK]. [SEP]'   
                prompt_3 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es de naturaleza [MASK] [SEP]'
                prompt_4 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es de carácter [MASK] [SEP]'
                prompt_5_1 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[0]} y {entity_names[1]}? La relación es el [MASK] [SEP]'
                prompt_5_2 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[0]} y {entity_names[1]}? La relación es la [MASK] [SEP]'    
   
                prompt_0_list_2_ent.append(prompt_0)
                prompt_1_list_2_ent.append(prompt_1)
                prompt_2_list_2_ent.append(prompt_2)
                prompt_0_list_2_ent.append(prompt_0)
                prompt_1_list_2_ent.append(prompt_1)
                prompt_2_list_2_ent.append(prompt_2)
                prompt_3_list_2_ent.append(prompt_3)
                prompt_4_list_2_ent.append(prompt_4)
                prompt_5_list_2_ent.append(prompt_5_1)
                prompt_5_list_2_ent.append(prompt_5_2)
            
            prompt_dict['prompt_0_ent_2'] = prompt_0_list_2_ent
            prompt_dict['prompt_1_ent_2'] = prompt_1_list_2_ent 
            prompt_dict['prompt_2_ent_2'] = prompt_2_list_2_ent
            prompt_dict['prompt_3_ent_2'] = prompt_3_list_2_ent
            prompt_dict['prompt_4_ent_2'] = prompt_4_list_2_ent
            prompt_dict['prompt_5_ent_2'] = prompt_5_list_2_ent

                

        elif entity_number in entity_dictionary and entity_number == 3:
                
                phrases = entity_dictionary.get(entity_number, [])


                prompt_0_list_3_ent = []
                prompt_1_list_3_ent  = []
                prompt_2_list_3_ent = []
                prompt_3_list_3_ent = []
                prompt_4_list_3_ent = []
                prompt_5_list_3_ent = []   

                for phrase in phrases:

                    entities = re.findall(r'\$(.*?)\$', phrase)
                    entity_names = [entity.strip() for entity in entities]
                    cleaned_phrase = re.sub(r'\$', '', phrase)

                    prompt_0_0 = f'[CLS] {cleaned_phrase} {entity_names[0]} [MASK] {entity_names[1]}. [SEP]'
                    prompt_0_1 = f'[CLS] {cleaned_phrase} {entity_names[1]} [MASK] {entity_names[2]}. [SEP]'
                    prompt_0_2 = f'[CLS] {cleaned_phrase} {entity_names[0]} [MASK] {entity_names[2]}. [SEP]'
                        
                    prompt_1_0 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es una relación de tipo [MASK]. [SEP]'
                    prompt_1_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[1]} y {entity_names[2]} es una relación de tipo [MASK]. [SEP]'
                    prompt_1_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[2]} y {entity_names[0]} es una relación de tipo [MASK]. [SEP]'

                    prompt_2_0 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es una relación de [MASK]. [SEP]'
                    prompt_2_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[1]} y {entity_names[2]} es una relación de [MASK]. [SEP]'
                    prompt_2_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[2]} y {entity_names[0]} es una relación de [MASK]. [SEP]'
                    
                    prompt_3_0 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es de naturaleza [MASK]. [SEP]'
                    prompt_3_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[1]} y {entity_names[2]} es de naturaleza [MASK]. [SEP]'
                    prompt_3_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[2]} y {entity_names[0]} es de naturaleza [MASK]. [SEP]'

                    prompt_4_0 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[0]} y {entity_names[1]} es de carácter [MASK]. [SEP]'
                    prompt_4_1 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[1]} y {entity_names[2]} es de carácter [MASK]. [SEP]'
                    prompt_4_2 = f'[CLS] {cleaned_phrase} La relación entre {entity_names[2]} y {entity_names[0]} es de carácter [MASK]. [SEP]'

                    prompt_5_1_0 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[0]} y {entity_names[1]}? La relación es el [MASK]. [SEP]'
                    prompt_5_1_1 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[1]} y {entity_names[2]}? La relación es el [MASK]. [SEP]'
                    prompt_5_1_2 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[2]} y {entity_names[0]}? La relación es el [MASK]. [SEP]'
                    prompt_5_2_0 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[0]} y {entity_names[1]}? La relación es la [MASK]. [SEP]'
                    prompt_5_2_1 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[1]} y {entity_names[2]}? La relación es la [MASK]. [SEP]'
                    prompt_5_2_2 = f'[CLS] {cleaned_phrase} ¿Cuál es la relación entre {entity_names[2]} y {entity_names[0]}? La relación es la [MASK]. [SEP]'
                    
                    prompt_0_list_3_ent.append(prompt_0_0)
                    prompt_0_list_3_ent.append(prompt_0_1)
                    prompt_0_list_3_ent.append(prompt_0_2)

                    prompt_1_list_3_ent.append(prompt_1_0)
                    prompt_1_list_3_ent.append(prompt_1_1)
                    prompt_1_list_3_ent.append(prompt_1_2)

                    prompt_2_list_3_ent.append(prompt_2_0)
                    prompt_2_list_3_ent.append(prompt_2_1)  
                    prompt_2_list_3_ent.append(prompt_2_2) 

                    prompt_3_list_3_ent.append(prompt_3_0)
                    prompt_3_list_3_ent.append(prompt_3_1)  
                    prompt_3_list_3_ent.append(prompt_3_2) 

                    prompt_4_list_3_ent.append(prompt_4_0)
                    prompt_4_list_3_ent.append(prompt_4_1)  
                    prompt_4_list_3_ent.append(prompt_4_2) 

                    prompt_5_list_3_ent.append(prompt_5_1_0)
                    prompt_5_list_3_ent.append(prompt_5_1_1)  
                    prompt_5_list_3_ent.append(prompt_5_1_2) 
                    prompt_5_list_3_ent.append(prompt_5_2_0)
                    prompt_5_list_3_ent.append(prompt_5_2_1)  
                    prompt_5_list_3_ent.append(prompt_5_2_2) 

                prompt_dict['prompt_0_ent_3'] = prompt_0_list_3_ent
                prompt_dict['prompt_1_ent_3'] = prompt_1_list_3_ent
                prompt_dict['prompt_2_ent_3'] = prompt_2_list_3_ent
                prompt_dict['prompt_3_ent_3'] = prompt_3_list_3_ent
                prompt_dict['prompt_4_ent_3'] = prompt_4_list_3_ent
                prompt_dict['prompt_5_ent_3'] = prompt_5_list_3_ent


        elif entity_number in entity_dictionary and entity_number == 1:

            phrases = entity_dictionary.get(entity_number, [])
                    
            prompt_list_unique = []
            prompt_list_unique_1 = []    
            for phrase in phrases:

                entities = re.findall(r'\$(.*?)\$', phrase)
                entity_names = [str(entity.strip()) for entity in entities]
                cleaned_phrase = re.sub(r'\$', '', phrase)
                        

                        
                prompt_0_unique = f"[CLS] {cleaned_phrase} La relación entre {''.join(entity_names)} y la frase anterior es una relación de [MASK]. [SEP]"
                prompt_1_unique= f"[CLS] {cleaned_phrase} La relación entre {''.join(entity_names)} y la frase anterior es una relación de tipo [MASK]. [SEP]"

                prompt_list_unique.append(prompt_0_unique)        
                prompt_list_unique_1.append(prompt_1_unique)

                prompt_dict['prompt_0_unique'] = prompt_list_unique
                prompt_dict['prompt_1_unique'] = prompt_list_unique_1

        if json_file_path_name:

            with open(json_file_path_name, 'w', encoding='utf-8') as json_file:
                    json.dump(prompt_dict, json_file)

        return prompt_dict




