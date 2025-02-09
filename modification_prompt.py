import copy
import ast
from itertools import permutations
from rdkit import Chem
from syntheseus import Molecule
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from multiprocessing import Pool, cpu_count
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from utils import *

def process_reaction_routes(route:list):
    json_list = []
    for idx,i in enumerate(route):
        products, reactants = i.split(">>")
        reaction = i
        reactants_list = reactants.split(".")
        if idx == 0:
            step = {
                "Molecule set": str([products]),
                "Product": str([products]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(reactants_list)
            }
            json_list.append(step)
        else:
            #print(json_list)
            original_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            update_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            #print(update_set)
            try:
                update_set.remove(products)
            except:
                print(update_set)
                print(products)
            update_set = update_set + reactants_list
            step = {
                "Molecule set": str(original_set),
                "Product": str([reaction]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(update_set)
            }
            json_list.append(step)
    
    return json_list

def check_availability(smi_list, inventory):
    unavailable_list = []
    for smi in smi_list:
        signal = inventory.is_purchasable(Molecule(smi))
        if not signal:
            unavailable_list.append(smi)
    
    return unavailable_list
    

def retrieve_routes(target_smi, all_fps, route_list, number):
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    fp = getfp(target_smi)
    sim_score = similarity_metric(fp, [fp_ for fp_ in all_fps])

    rag_tuples = list(zip(sim_score, route_list))
    rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:50]

    route_list = [t[1] for t in rag_tuples]
    sims_list = [t[0] for t in rag_tuples]


    sum_scores = sum(sims_list)
    population_probs = [p / sum_scores for p in sims_list]
    sampled_index = np.random.choice(len(route_list), p=population_probs, size=3, replace=False)
    sampled_routes = [route_list[i] for i in sampled_index]

    return sampled_routes

def modification_hints(unperchasable_molecule, all_fps, route_list):

    fb = '''\n
    '''
    for smi in unperchasable_molecule:
        examples = retrieve_routes(smi, all_fps, route_list, 3)
        for i in examples:
            fb = fb + str(process_reaction_routes(i)) + '\n'

    return fb

def construct_modification_prompt(molecule_smi_list, examples, evaluation, inventory):
    initial_prompt = '''
    As a professional chemist specialized in synthesis analysis, you are tasked with proposing or modifying full retrosynthesis route for target molecules provided in SMILES format.
    A retrosynthesis route is a series of retrosynthesis steps that starts from the given target molecule set and ends with some commercially purchasable compounds. In the route, you need to keep a molecule set, which are the molecules we need. 
    At the first step, the molecule set should be the target molecule set given by the user. In each step, you need to provide a backward reaction and update this molecule set. Specifically, you need to remove the product molecule of the reaction from the molecule set and then add the reactants into it.
    By doing so, you will end with a molecule set in which all the molecules are commercially purchasable. The reactions are from the USPTO dataset. Please also take reactions in stereochemistry into consideration. For example, E-configuration or Z-configuration.
    '''
    previous_route, feedback = get_feedback(molecule_smi_list, evaluation, inventory)
    task_description = '''
    My target molecule set is: {}

    In the previous attempt, the first step is: 
    <ROUTE>
    {}
    </ROUTE>
    The feedback for this step is: {}


    Meanwhile, to assist you, example retrosynthesis routes that are close to the molecules in the starting molecule set will be provided.\n {}

    Please propose a retrosynthesis route for the starting molecule set. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge. All the molecules should be in SMILES format. For example, Cl2 should be ClCl in SMILES format. Br2 should be BrBr in SMILES format. H2O should be O in SMILES format. HBr should be [H]Br in SMILES format. NH3 should be N in SMILES format. Hydrogen atoms are implicitly understood unless explicitly needed for clarity.
    '''.format(str(molecule_smi_list), previous_route, feedback, examples)

    requirements = '''
    The route should be a list of steps wrapped in <ROUTE></ROUTE> with <EXPLAINATION></EXPLAINATION> after it. Each step in the list should be a dictionary. You need to keep a molecule set in which are the molecules we need to synthesize or purchase. In each step, you need to select a molecule from the 'Molecule set'
    as the prodcut molecule in this step and use a backward reaction to find the reactants. After taking the backward reaction in this step, you need to remove the product molecule from the molecule set and add the reactants you find into the molecule set and then name
    this updated set as the 'Updated molecule set' in this step. In the next step, the starting molecule set should be the 'Updated molecule set' from the previous step. In the last step, all the molecules in the 'Updated molecule set' should be purchasable. Here is an example:
    corresponds to a set of molecules that are commercially available. Here is an example:

    <ROUTE>
    [   
        {
            'Molecule set': "[Target molecules]",
            'Rational': Step analysis,
            'Product': "[Product molecule]",
            'Reaction': "[Reaction template]",
            'Reactants': "[Reactant1, Reactant2]",
            'Updated molecule set': "[Reactant1, Reactant2]"
        },
        {
            'Molecule set': "[Reactant1, Reactant2]",
            'Rational': Step analysis,
            'Product': "[Product molecule]",
            'Reaction': "[Reaction template]",
            'Reactants': "[subReactant1, subReactant2]",
            'Updated molecule set': "[Reactant1, subReactant1, subReactant2]"
        }
    ]
    </ROUTE>
    <EXPLAINATION>: Explaination for the whole route. </EXPLAINATION>
    \n\n
    Requirements: 1. The 'Molecule set' contains all of the molecules we need to synthesize. In the first step, it should be the list of target molecules given by the user. In the following steps, it should be the 'Updated molecule set' from the previous step.\n
    2. The 'Rational' part in each step should be your analysis for syhthesis planning in this step. It should be in the string format wrapped with \'\'\n
    3. 'Product' is the molecule we plan to synthesize in this step. It should be from the 'Molecule set'. The molecule should be a molecule from the 'Molecule set' in a list. The molecule smiles should be wrapped with \'\'.\n
    4. 'Reaction' is a reaction which can synthesize the product molecule. It should be in a list. The reaction template should be in SMILES format. For example, [Product>>Reactant1.Reactant2].\n
    5. 'Reactants' are the reactants of the reaction. It should be in a list. The molecule smiles should be wrapped with \'\'.\n
    6. The 'Updated molecule set' should be molecules we need to purchase or synthesize after taking this reaction. To get the 'Updated molecule set', you need to remove the product molecule from the 'Molecule set' and then add the reactants in this step into it. In the last step, all the molecules in the 'Updated molecule set' should be purchasable.\n
    7. In the <EXPLAINATION>, you should analyze the whole route and ensure the molecules in the 'Updated molecule set' in the last step are all purchasable.\n
    '''

    question = initial_prompt + requirements + task_description

    return question


def get_feedback(molecule_set, evaluation, inventory):
    original_set = copy.deepcopy(molecule_set)
    update_set = copy.deepcopy(original_set)
    try:
        update_set.remove(evaluation['product'][0])
    except:
        update_set = []
    update_set = update_set + evaluation['reactants']     
    step = [{
            "Molecule set": str(molecule_set),
            "Product": str(evaluation['product']),
            "Reaction": str([evaluation['reaction']]),
            "Reactants": str(evaluation['reactants']),
            "Updated molecule set": str(update_set)
        }]

    feedback = '''\n'''
    # reaction not exist
    if evaluation['reaction_existance'] == False:
        feedback += reaction_unavailable_feedback(evaluation)
                            
    # molecule in valid
    if len(evaluation['invalid_updated_mol_id']) != 0:
        feedback += molecule_invalid_feedback(evaluation)

    if evaluation['reaction_existance'] == True and evaluation['reaction_valid'] == False:
        feedback += reaction_can_not_happen(evaluation)
    
    # runreaction error
    if evaluation['reaction_existance'] == True and evaluation['reaction_valid'] == True and evaluation['updated_set_valid'] == False:
        feedback += updated_set_mismatch_feedback(evaluation)

        
    # molecule unavailable
    if evaluation['check_availability'] == True and len(evaluation['unavailable_mol_id']) != 0 and len(evaluation['invalid_updated_mol_id']) == 0:
        feedback += molecule_unavailable_feedback(evaluation, inventory)

    
    if feedback == '''\n''':
        feedback = 'The provided route is valid. Please try to propose a new synthetic route for this target molecule.'
                


    return str(step), feedback