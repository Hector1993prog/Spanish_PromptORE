from PromptORE import pipeline_PromptORE
import click

@click.command()
@click.option("--xml_file", help="Path to the input XML file", required=True)
@click.option("--models_path", help="Path to the models directory", required=True)
@click.option("--output_dir", help="Path to the models output directory for CSV files", required=True)
@click.option("--batch_size", type=int, default=32, help="number of sentences per batch")
@click.option("--entity_number", type=int, default=10, help="number of entities to extract with the method")
def main(xml_file, models_path, output_dir, batch_size, entity_number):
    pipeline = pipeline_PromptORE()

    pipeline.run_models(
        xml_input=xml_file,
        models_path=models_path,
        output_folder=output_dir,
        batch_size=batch_size,
        entity_number=entity_number
    )

if __name__ == "__main__":
    main()
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

    batch_size=batch_size,
    entity_number=entity_number
)

