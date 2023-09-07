from Ro_prompt_generator import phrase_extraction, prompt_generator
from Ro_prompt_preprocessing import tokenize_prompts_Roberta
from Ro_inference import extract_Roberta_embeddings_dataframe
from Ro_clustering import compute_kmeans_clustering, plot_elbow_curve
import pandas as pd
import os


class pipeline_PromptORE_Roberta:
    def __init__(self):
        pass

    """
    A pipeline class for generating prompts, extracting RoBERTa embeddings, and performing clustering using PromptORE approach.

    Args:
        None

    Attributes:
        None

    Methods:
        run_pipeline(xml_input, model_name, batch_size, entity_number, prompt_type, json_file_path_name, max_k) -> pd.DataFrame:
            Runs the complete pipeline for generating prompts, extracting BERT embeddings, and performing clustering.

    Usage:
        pipeline = pipeline_PromptORE()
        results_dataframe = pipeline.run_pipeline(xml_input='input.xml', model_name='model_name', batch_size=8, entity_number=2, prompt_type='prompt_1', json_file_path_name='output.json', max_k=20)
    """

    def run_pipeline_base(
        self,
        xml_input: str = None,
        model_name: str = 'PlanTL-GOB-ES/roberta-base-bne',
        batch_size: int = 8,
        full_prompt: bool = True,
        entity_number: int = 2,
        prompt_type: str = 'prompt_1_ent_2',
        full_extraction: bool = False,
        json_file_path_name: str = 'prompts_roberta.json',
        elbow_curve: bool = False,
        num_clusters:int = 4,
        max_k: int = 20,
        return_tensors: bool = False,
        save_df: bool = False
    ) -> pd.DataFrame:

        """
        Runs the complete pipeline for generating prompts, extracting BERT embeddings,
        and performing clustering using the PromptORE approach.

        Args:
            xml_input (str, mandatory): The path to the XML input file. Default is None.
            model_name (str, optional): The name of the BERT model to use. Default is 'PlanTL-GOB-ES/roberta-large-bne'.
            batch_size (int, optional): Batch size for BERT embeddings extraction. Default is 8.
            full_prompt (bool, optional): if true returns a json file with all the prompts.
            entity_number (int, optional): The number of entities for prompt generation. Default is 2.
            prompt_type (str, optional): The type of prompt to use. Default is 'prompt_1'.
            full_extraction (bool, optional): Overwrite entity_number taking all the possible prompts within a specific number of entities. Default is False.
            json_file_path_name (str, optional): The path to save the generated prompts in JSON format. Default is None.
            max_k (int, optional): The maximum number of clusters to consider for elbow curve. Default is 20.
            return_tensors (bool, optional): If True return the pd.DataFrame with the embeddings attached for the MASK token. Default is False.
            save_df (bool, optional): Parameter if you want to save the results. Default is False

        Returns:
            pd.DataFrame: A DataFrame containing the first 10 predicted tokens,
            objective phrases, BERT embeddings, and predicted clustering labels.

        Usage:
            pipeline = PipelinePromptORE()
            results_dataframe = pipeline.run_pipeline(xml_input='input.xml', model_name='model_name',
                                                     batch_size=8, entity_number=2, prompt_type='prompt_1',
                                                     json_file_path_name='output.json', max_k=20)
        """
        if xml_input is None:
            raise ValueError("xml file is required")
        
        model_size = 'base'

        entity_dictionary = phrase_extraction(xml_path=xml_input, return_dict=True)
        print('Phrases with entities extracted')

        prompt_dictionary = prompt_generator(
            phrase_input=entity_dictionary,
            entity_number=entity_number,
            json_file_path_name=json_file_path_name,
            full_extraction=full_prompt
        )
        print('Prompts generated')

        tokenized_inputs = tokenize_prompts_Roberta(
            prompt_dict = prompt_dictionary,
            model_size = model_size,
            prompt_type = prompt_type,
            full_extraction = full_extraction
            
        )
        print('Inputs tokenized')

        embeddings_dataframe = extract_Roberta_embeddings_dataframe(
            inputs_tokenized=tokenized_inputs,
            model_name=model_name,
            model_size= model_size,
            batch_size=batch_size
        )
        if elbow_curve:
            plot_elbow_curve(embeddings_dataframe, max_k=max_k)

            num_clusters = int(input("Enter the number of clusters: "))

            clusters = compute_kmeans_clustering(
                embeddings_dataframe,
                n_rel=num_clusters,
                random_state=42
            )
        else:
            num_clusters = num_clusters
            clusters = compute_kmeans_clustering(
                embeddings_dataframe,
                n_rel=num_clusters,
                random_state=42
            )  
            
        embeddings_dataframe['predicted_label'] = clusters
        clusters_dataframe = embeddings_dataframe
        if return_tensors:
            clusters_dataframe = clusters_dataframe

        else:
            col_embed = 'mask_embedding'
            clusters_dataframe = clusters_dataframe.drop(col_embed, axis=1)
        if save_df:
            clusters_dataframe.to_csv('clusters_dataframe.csv', encoding='utf-8')
        print('DataFrame created')
        return clusters_dataframe


    def run_models_base(self,
                xml_input:str = None,
                models_path: str = None,
                output_folder: str = None,
                entity_number: int = 6,
                batch_size: int = 32
                ):
        """
        Runs multiple models using a pipeline and saves the results to CSV files.

        Args:
            xml_input (str): The path to the XML input file.
            models_path (str): The path to the models.
            output_folder (str): The path to the folder where the CSV files will be stored.
        """
        models_dict = {}
        print('Running program')
        print(f'xml input at {xml_input}')
        for root, dirs, files in os.walk(models_path):
            for dire in dirs:
                if dire != '__pycache__' and dire != '__results__files':
                    models_dict[dire] = os.path.join(root, dire)

        for model, path in models_dict.items():
            print(f'running model: {model}')
            

            results = self.run_pipeline_base(
                xml_input=xml_input,
                model_name = path,
                batch_size=batch_size,
                entity_number=entity_number,
                full_extraction=True,
                elbow_curve=False,
                max_k=10,
                save_df=False,
                num_clusters=6
            )

            # Create the output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            results.to_csv(os.path.join(output_folder, f'{model}_clustering.csv'))
            print(f'data set {model}_clustering.csv created at {output_folder}')

        print('Program Completed')

    def run_pipeline_large(
        self,
        xml_input: str = None,
        model_name: str = 'PlanTL-GOB-ES/roberta-large-bne',
        batch_size: int = 8,
        full_prompt: bool = True,
        entity_number: int = 2,
        prompt_type: str = 'prompt_1_ent_2',
        full_extraction: bool = False,
        json_file_path_name: str = 'prompts_roberta.json',
        elbow_curve: bool = False,
        num_clusters:int = 4,
        max_k: int = 20,
        return_tensors: bool = False,
        save_df: bool = False
    ) -> pd.DataFrame:

        """
        Runs the complete pipeline for generating prompts, extracting BERT embeddings,
        and performing clustering using the PromptORE approach.

        Args:
            xml_input (str, mandatory): The path to the XML input file. Default is None.
            model_name (str, optional): The name of the BERT model to use. Default is 'PlanTL-GOB-ES/roberta-large-bne'.
            batch_size (int, optional): Batch size for BERT embeddings extraction. Default is 8.
            full_prompt (bool, optional): if true returns a json file with all the prompts.
            entity_number (int, optional): The number of entities for prompt generation. Default is 2.
            prompt_type (str, optional): The type of prompt to use. Default is 'prompt_1'.
            full_extraction (bool, optional): Overwrite entity_number taking all the possible prompts within a specific number of entities. Default is False.
            json_file_path_name (str, optional): The path to save the generated prompts in JSON format. Default is None.
            max_k (int, optional): The maximum number of clusters to consider for elbow curve. Default is 20.
            return_tensors (bool, optional): If True return the pd.DataFrame with the embeddings attached for the MASK token. Default is False.
            save_df (bool, optional): Parameter if you want to save the results. Default is False

        Returns:
            pd.DataFrame: A DataFrame containing the first 10 predicted tokens,
            objective phrases, BERT embeddings, and predicted clustering labels.

        Usage:
            pipeline = PipelinePromptORE()
            results_dataframe = pipeline.run_pipeline(xml_input='input.xml', model_name='model_name',
                                                     batch_size=8, entity_number=2, prompt_type='prompt_1',
                                                     json_file_path_name='output.json', max_k=20)
        """
        if xml_input is None:
            raise ValueError("xml file is required")
        model_size = 'large'
        entity_dictionary = phrase_extraction(xml_path=xml_input, return_dict=True)
        print('Phrases with entities extracted')

        prompt_dictionary = prompt_generator(
            phrase_input=entity_dictionary,
            entity_number=entity_number,
            json_file_path_name=json_file_path_name,
            full_extraction=full_prompt
        )
        print('Prompts generated')

        tokenized_inputs = tokenize_prompts_Roberta(
            prompt_dict = prompt_dictionary,
            model_size = model_size,
            prompt_type = prompt_type,
            full_extraction = full_extraction
            
        )
        print('Inputs tokenized')

        embeddings_dataframe = extract_Roberta_embeddings_dataframe(
            inputs_tokenized=tokenized_inputs,
            model_name=model_name,
            model_size= model_size,
            batch_size=batch_size
        )
        if elbow_curve:
            plot_elbow_curve(embeddings_dataframe, max_k=max_k)

            num_clusters = int(input("Enter the number of clusters: "))

            clusters = compute_kmeans_clustering(
                embeddings_dataframe,
                n_rel=num_clusters,
                random_state=42
            )
        else:
            num_clusters = num_clusters
            clusters = compute_kmeans_clustering(
                embeddings_dataframe,
                n_rel=num_clusters,
                random_state=42
            )  
            
        embeddings_dataframe['predicted_label'] = clusters
        clusters_dataframe = embeddings_dataframe
        if return_tensors:
            clusters_dataframe = clusters_dataframe

        else:
            col_embed = 'mask_embedding'
            clusters_dataframe = clusters_dataframe.drop(col_embed, axis=1)
        if save_df:
            clusters_dataframe.to_csv('clusters_dataframe.csv', encoding='utf-8')
        print('DataFrame created')
        return clusters_dataframe


    def run_models_large(self,
                xml_input:str = None,
                models_path: str = None,
                output_folder: str = None,
                entity_number: int = 6,
                batch_size: int = 32
                ):
        """
        Runs multiple models using a pipeline and saves the results to CSV files.

        Args:
            xml_input (str): The path to the XML input file.
            models_path (str): The path to the models.
            output_folder (str): The path to the folder where the CSV files will be stored.
        """
        models_dict = {}
        model_size = 'large'
        print('Running program')
        print(f'xml input at {xml_input}')
        for root, dirs, files in os.walk(models_path):
            for dire in dirs:
                if dire != '__pycache__' and dire != '__results__files':
                    models_dict[dire] = os.path.join(root, dire)

        for model, path in models_dict.items():
            print(f'running model: {model}')
            

            results = self.run_pipeline_large(
                xml_input=xml_input,
                model_size = model_size,
                model_name = path,
                batch_size=batch_size,
                entity_number=entity_number,
                full_extraction=True,
                elbow_curve=False,
                max_k=10,
                save_df=False,
                num_clusters=6
            )

            # Create the output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            results.to_csv(os.path.join(output_folder, f'{model}_clustering.csv'))
            print(f'data set {model}_clustering.csv created at {output_folder}')

        print('Program Completed')    