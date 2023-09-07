from Ro_promptORE import pipeline_PromptORE_Roberta
import argparse
import os

parser = argparse.ArgumentParser(description="Runs the complete pipeline for generating prompts, extracting RoBERTa embeddings,\
        and performing clustering using the PromptORE approach and based for xml-tie files.")

parser.add_argument("--xml_file", help="Path to the input XML file", required=True)
parser.add_argument("--models_path", help="Path to the models directory", required=True)
parser.add_argument("--output_dir", help="Path to the models output directory for CSV files", required=True)
parser.add_argument("--model_size", type=str, default='base', help='Model size for roberta. it handles base and large')
parser.add_argument("--entity_number", type=int, default=3, help="number of entities to extract with the method")
parser.add_argument("--batch_size", type=int, default=32, help="number of sentences per batch")

args = parser.parse_args()

xml_file = args.xml_file
models_path = args.models_path
output_dir = args.output_dir 
model_size = args.model_size
entity_number = args.entity_number
batch_size =  args.batch_size
# Check if the XML file exists
if not os.path.isfile(xml_file):
    print(f"Error: The XML file '{xml_file}' does not exist.")
    exit(1)

pipeline = pipeline_PromptORE_Roberta()

if model_size == 'base':

    pipeline.run_models_base(
                xml_input = xml_file,
                models_path = models_path,
                output_folder = output_dir,
                entity_number = entity_number,
                batch_size = batch_size
                )
elif model_size == 'large':
    pipeline.run_models_large(
                xml_input = xml_file,
                models_path = models_path,
                output_folder = output_dir,
                entity_number = entity_number,
                batch_size = batch_size
                )
else:
    print('Error: Please introduce "large" or "base" to perform the extraction.')
    exit(1)
