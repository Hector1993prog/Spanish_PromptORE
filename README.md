# Spanish_PromptORE

This repository offers an implementation of the paper PromptORE - A Novel Approach Towards Fully Unsupervised Relation Extraction.

Here we present a class based on transformers and torch to perform the full experiment of the paper based on. It was adapted to spanish text as well as XML-TIE inputs. The class will yield a dataframe with the relations extracted without supervision. It is though to being executed over a sample (small to facilitated the labelling) and then classify manually. Another option is to use the prompt_type desired and execute the k-means clustering over one of the prompts.



## in short, the model allows you to extract relations and classify then based on XML-TIE inputs with already extracted entities.



The code is based on a XML input to export a CSV file with a set of words predicted by the transformer model. You need a XML-TIE input to make it work.





The model was though to work with Spanish based texts. Therefore, to use it with another language, you need to change the files related to tokenization, BERT or RoBERTa models as well as the promp_generator.py file.



Further description of the model and a jupiternotebook of demonstration will be added as well as the requirements for the virtual environment.
