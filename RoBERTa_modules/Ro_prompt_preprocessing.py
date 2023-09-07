from transformers import AutoTokenizer

def tokenize_prompts_Roberta(prompt_dict: dict,
                             model_size: str = 'base',
                             prompt_type: str = None,
                             full_extraction: bool = False
                             ) -> dict:
    """Compute PromptORE relation embedding for the list extracted from the json format.

    Args:
        prompt_dict (dict): The dictionary from where to extract the prompts.
        prompt_type (str): The type of prompt to analyse. Default: none.
        full_extraction (bool): If True, a list appends all the prompts in one list.
    Returns:
        dict: inputs dictionary of the model"""
    if prompt_dict is None:
        raise ValueError('prompt_dictionary is required')

    if not full_extraction and prompt_type is None:
        raise ValueError('If full_extraction is not intended, please introduce a prompt_type')

    tokenizer = AutoTokenizer.from_pretrained(f'PlanTL-GOB-ES/roberta-{model_size}-bne')
    prompt_type_list = []

    if full_extraction:
        all_phrases = [phrase for value_list in prompt_dict.values() for phrase in value_list]
        inputs = tokenizer.batch_encode_plus(
            all_phrases,
            max_length=512,
            truncation=True,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False
        )
    else:
        if prompt_type in prompt_dict:
            prompt_type_list.extend(prompt_dict[prompt_type])

        inputs = tokenizer.batch_encode_plus(
            prompt_type_list,
            max_length=512,
            truncation=True,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False
        )

    return inputs

