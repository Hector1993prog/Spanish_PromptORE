import torch
import pandas as pd
from tqdm import tqdm  # tqdm for progress tracking
from transformers import BertForMaskedLM, BertTokenizer
from prompt_preprocessing import tokenize_prompts_beto

class MeditationsDataSet(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return self.encodings.input_ids.shape[0]
    
    def __getitem__(self, index):
        return {key: val[index] for key, val in self.encodings.items()}

def extract_bert_embeddings_dataframe(
        inputs_tokenized= None,
        model_name: str = 'dccuchile/bert-base-spanish-wwm-uncased',
        tokenizer_name: str = 'dccuchile/bert-base-spanish-wwm-uncased',
        batch_size: int = 8,
        
        ) -> pd.DataFrame:
    """
    Extracts BERT embeddings for masked tokens in the context of phrases from a JSON file.
    
    Args:
        inputs_tokenized (dict): toeknized imputs for beto.
        model_name (str): Name of the BERT model to use. Default: 'dccuchile/bert-base-spanish-wwm-uncased'
        tokenizer_name (str): Name of the BERT tokenizer to use. Default: 'dccuchile/bert-base-spanish-wwm-uncased'.
        prompt_type (str): The type of prompt that the tokenize_prompts_beto processes. Default: "prompt_1".
        full_extraction: (bool): This parameter returns a full list for all the prompts in the json file.
        batch_size (int): Batch size for processing.
        
    Returns:
        pd.DataFrame: A DataFrame containing predicted tokens, mask predictions, and mask embeddings.
    """
    if inputs_tokenized is None:
        raise ValueError("inputs_tokenized is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertForMaskedLM.from_pretrained(model_name).to(device)

    inputs = inputs_tokenized
    
    predicted_tokens_list = []
    mask_predictions = []
    mask_embeddings = []
    
    dataset = MeditationsDataSet(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Use tqdm instead of tqdm_notebook for progress tracking
    loop = tqdm(loader, leave=True)

    for batch in loop:
        model.eval()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():  # No need for gradient during evaluation
            outputs = model(input_ids, attention_mask = attention_mask)[0]

        # Assuming you want the last layer's hidden state
        hidden_states = outputs

        for batch_idx in range(len(outputs)):
            input_ids_batch = input_ids[batch_idx]  # Get input_ids for the current batch
            mask_positions = torch.where(input_ids_batch == tokenizer.mask_token_id)[0]  # Find mask positions
            
            for mask_idx in mask_positions:
                idxs = torch.argsort(outputs[batch_idx, mask_idx], descending=True)
                masked_embedding = hidden_states[batch_idx, mask_idx].tolist()
                predicted_token_ids = idxs[:9]
                predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)
                mask_prediction = tokenizer.decode(input_ids_batch.tolist(), skip_special_tokens=True)  # Original input phrase
                mask_prediction = mask_prediction.replace(tokenizer.mask_token, predicted_tokens[0])  # Replace [MASK] with predicted token
                
                predicted_tokens_list.append(predicted_tokens)
                mask_predictions.append(mask_prediction)
                mask_embeddings.append(masked_embedding)

    df = pd.DataFrame({
            'predicted_token': predicted_tokens_list,
            'predicted_phrase': mask_predictions,
            'mask_embedding': mask_embeddings
            })

    return df




