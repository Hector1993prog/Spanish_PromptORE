from PromptORE import pipeline_PromptORE
import argparse
import os

parser = argparse.ArgumentParser(description="Runs the complete pipeline for generating prompts, extracting BERT embeddings,\
        and performing clustering using the PromptORE approach.")

parser.add_argument("--xml_file", help="Path to the input XML file", required=True)
parser.add_argument("--models_path", help="Path to the models directory", required=True)
parser.add_argument("--output_dir", help="Path to the models output directory for CSV files", required=True)
parser.add_argument("--batch_size", type=int, default=32, help="number of sentences per batch")
parser.add_argument("--entity_number", type=int, default=10, help="number of entities to extract with the method")

args = parser.parse_args()

xml_file = args.xml_file
models_path = args.models_path
output_dir = args.output_dir 
batch_size =  args.batch_size
entity_number = args.entity_number

# Check if the XML file exists
if not os.path.isfile(xml_file):
    print(f"Error: The XML file '{xml_file}' does not exist.")
    exit(1)

pipeline = pipeline_PromptORE()

pipeline.run_models(
    xml_input=xml_file,
    models_path=models_path,
    output_folder=output_dir,
    batch_size=batch_size,
    entity_number=entity_number
)

