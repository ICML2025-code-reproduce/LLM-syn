# LLM-syn

#### 1. Setup the environment

##### (1) Download the repository
    
    git clone https://github.com/ICML2025-code-reproduce/LLM-syn.git
    cd LLM-syn
    
##### (2) Create a conda environment
    
    conda create -n llmsyn python=3.10
    conda activate llmsyn
    pip install openai == 0.28.1
    pip install "syntheseus[all]"
    pip install rdchiral

#### 2. Download the data and set up the API key

##### (1) Download the building block molecules 

Download and unzip the files from this [link](https://www.dropbox.com/scl/fi/6qcv3bg9ka7x4cf2vci3v/inventory.zip?rlkey=f22o1iu44ye0w8geyyzna6zop&st=c0ecyetp&dl=0), 
and put inventory.pkl under the ```dataset``` directory.

##### (2) Download the the SCScore model
    
    cd LLM-syn
    git clone https://github.com/connorcoley/scscore.git

##### (3) set up the API key

Set up the OpenAI API key in the ```query_LLM``` function in the ```optimizer.py```.

#### 3. Example usage

To plan with LLM-syn-planner on USPTO Easy dataset, run the following command,

    mkdir USPTO-easy
    mkdir found_route
    python main.py --method planning --dataset_name USPTO-easy