import os
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
import json
from rdkit import Chem
from syntheseus import Molecule
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions, DataStructs
from itertools import permutations
from utils import *
from scscore.scscore.standalone_model_numpy import *
import pickle
import openai
import json
from concurrent.futures import ThreadPoolExecutor
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)




def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

def preprocess_reaction_dict(reaction_dict):
    """Preprocess reaction_dict to compile SMARTS into RDKit Mol objects."""
    preprocessed_dict = {}
    for key, smarts in reaction_dict.items():
        try:
            products, reactants = smarts.split(">>")
            reactant_mols = [Chem.MolFromSmarts(r) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmarts(p) for p in products.split(".")]
            preprocessed_dict[key] = (reactant_mols, product_mols)
        except Exception as e:
            print(f"Error preprocessing SMARTS {smarts}: {e}")
    return preprocessed_dict

#Return SC score and store routes
class Oracle:
    def __init__(self, args=None, route_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.route_buffer = {} if route_buffer is None else route_buffer

        self.last_log = 0

        self.sc_Oracle = sc_oracle()

    def get_oracle_score(self, mol_smiles):
        smi, SC_score = self.sc_Oracle.get_score_from_smi(mol_smiles)

        return SC_score
    @property
    def budget(self):
        return self.max_oracle_calls
    
    def reward(self, inventory, updated_molecule_set:list):
        final_score = 0
        score_list = []
        for smi in updated_molecule_set:
            try:
                signal = inventory.is_purchasable(Molecule(smi))
                if not signal:
                    score_list.append(self.get_oracle_score(smi))
            except Exception as e:
                print(f"Error: {e}")
                score_list.append(5)
        score_sum = sum(score_list)
        if len(score_list) != 0:
            score_mean = score_sum / len(score_list) 
        else:
            score_mean = 0
        combined_score = score_mean + score_sum
        final_score = final_score - combined_score
        return final_score
    
    def evaluate(self, inventory, route_evaluation):
        #print(route_evaluation)
        for idx, step in enumerate(route_evaluation):
            #print(step)
            if step[1] == False:
                score = self.reward(inventory, step[2]['molecule_set'])
                return score
            elif step[1] == True:
                continue
        
        #last step
        print(route_evaluation[-1])
        if route_evaluation[-1][2]['check_availability'] == True and len(route_evaluation[-1][2]['unavailable_mol_id']) == 0:
            score = 0
            return score
        else:
            score = self.reward(inventory, route_evaluation[-1][2]['updated_molecule_set'])
            return score

    def sort_buffer(self):
        self.route_buffer = dict(sorted(self.route_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.route_buffer, f, sort_keys=False)

    def log_intermediate(self, finish=False):
        if finish:
            n_calls = self.max_oracle_calls
            self.save_result(self.task_label)

    def __len__(self):
        return len(self.route_buffer) 

    def score_route(self, inventory, route_evaluation):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.route_buffer) > self.max_oracle_calls:
            return -15
        if route_evaluation is None:
            return -15
        dict_key = json.dumps(route_evaluation)
        if dict_key in self.route_buffer:
            pass
        else:
            self.route_buffer[dict_key] = [float(self.evaluate(inventory, route_evaluation)), len(self.route_buffer)+1]
        return self.route_buffer[dict_key][0]
    
    def __call__(self, inventory, route_evaluation):
        """
        Score
        """
        score_list = self.score_route(inventory, route_evaluation)
        if len(self.route_buffer) % self.freq_log == 0 and len(self.route_buffer) > self.last_log:
            self.sort_buffer()
            self.last_log = len(self.route_buffer)
            self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.route_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args

        self.oracle = Oracle(args=self.args)

        args.template_path = 'dataset/idx2template_retro.json'
        args.inventory_path = 'dataset/inventory.pkl'
        self.original_template_dict = self.load_template(args.template_path)
        self.template_dict =  preprocess_reaction_dict(self.original_template_dict)
        self.inventory = self.load_inventory(args.inventory_path)
        self.reaction_list, self.all_reaction_fps = self.get_reaction_fps(self.original_template_dict)
        self.reaction_product_mols = [value[1][0] for key, value in self.template_dict.items()]
        self.explored_reaction = set()


    def load_template(self, template_path):
        with open(template_path, "r") as f:
            template_dict = json.load(f)
        #preprocessed_dict = preprocess_reaction_dict(template_dict)
        return template_dict

    
    def load_inventory(self, inventory_path):
        with open(inventory_path, 'rb') as file:
            inventory = pickle.load(file)
        
        return inventory
    
    def get_reaction_fps(self, template_dict):
        reaction_list = list(template_dict.values())
        getreactionfp = lambda smart_reaction: rdChemReactions.CreateDifferenceFingerprintForReaction(rdChemReactions.ReactionFromSmarts(smart_reaction))
        all_reaction_fps = []
        for reaction in reaction_list:
            all_reaction_fps.append(getreactionfp(reaction))
        
        return reaction_list, all_reaction_fps
    def sanitize_smiles(self, smiles):
        """
        Check if a SMILES string is valid and return the sanitized molecule.

        Parameters:
            smiles (str): SMILES string.

        Returns:
            str or None: Sanitized SMILES string if valid, None otherwise.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol)
            else:
                return None
        except Exception:
            return None

    def sanitize_reaction(self, reaction_smiles):
        """
        Process a reaction SMILES, removing invalid molecules in reactants and products.

        Parameters:
            reaction_smiles (str): Reaction SMILES in the format "reactants>>products".

        Returns:
            str: Sanitized reaction SMILES with only valid reactants and products.
        """
        try:
            reactants, products = reaction_smiles.split('>>')
        
            reactants_list = reactants.split('.')
            products_list = products.split('.')

            sanitized_reactants = [self.sanitize_smiles(smiles) for smiles in reactants_list]
            sanitized_products = [self.sanitize_smiles(smiles) for smiles in products_list]

            sanitized_reactants = [s for s in sanitized_reactants if s is not None]
            sanitized_products = [s for s in sanitized_products if s is not None]

            sanitized_reaction = ".".join(sanitized_reactants) + ">>" + ".".join(sanitized_products)

            return sanitized_reaction

        except Exception as e:
            print(f"Invalid reaction SMILES format: {reaction_smiles}. Error: {e}")
            return reaction_smiles

    def blurry_search(self, reaction_smiles, product_smiles):
        similarity_metric = DataStructs.BulkTanimotoSimilarity
        try:
            sanitized_reaction = self.sanitize_reaction(reaction_smiles)
            fp_re = rdChemReactions.CreateDifferenceFingerprintForReaction(smiles_to_reaction(sanitized_reaction))
            sims = similarity_metric(fp_re, [fp_ for fp_ in self.all_reaction_fps])
            rag_tuples = list(zip(sims, self.reaction_list))
            rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:100]
            sorted_reaction_list = [t[1] for t in rag_tuples]
            
        except Exception as e:
            print(f"Error {e} getting reaction {reaction_smiles} for fingerprints!")
            return self.blurry_search_from_product(reaction_smiles, product_smiles)
        
        for reaction_smarts in sorted_reaction_list:
            try:
                target_rd = rdchiralReactants(product_smiles)
                reaction_outputs = run_retro(target_rd, reaction_smarts)
                if len(reaction_outputs) > 1:
                    reaction_outputs = self.rank_reactants(reaction_outputs)
                if len(reaction_outputs) == 0:
                    continue
                reactants_generated = [reactant for reactant in reaction_outputs[0]]
                if reactants_generated == []:
                    continue
                elif len(reactants_generated) > 0:
                    keys = [key for key, value in self.original_template_dict.items() if value == reaction_smarts]
                    if (product_smiles, keys[0]) in self.explored_reaction:
                        continue
                    return True, keys[0], reaction_smarts
            except Exception as e:
                print(f"Error {e} testing reaction {reaction_smarts} on product {product_smiles}")
                continue
        return self.blurry_search_from_product(reaction_smiles, product_smiles)

    def blurry_search_from_product(self, reaction_smiles, product_smiles):

        return False, None, reaction_smiles

    
    def sanitize(self, starting_list, route):
        new_route = check_and_update_routes(route, starting_list)
        first_evaluation = self.check_route(starting_list, new_route)

        new_route = map_reaction(new_route, first_evaluation)
        new_route = self.fix_reaction_error(new_route, first_evaluation)
        new_route = check_and_update_routes(new_route, starting_list)

        final_evaluation = self.check_route_extra(starting_list, new_route, first_evaluation)

        return new_route, final_evaluation

    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, finish=False):
        self.oracle.log_intermediate(finish=finish)
    

        
    def save_result(self, suffix=None):

        print(f"Saving...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.route_buffer, f, sort_keys=False)
    
    def check_route(self, target_smi, route):
        """
        Check if the route is valid
        target smi is a list
        """
        results = []
        for i in range(len(route)):
            current_step_index = i
            current_step = route[current_step_index]

            step_validity = False
            molecule_set = ast.literal_eval(current_step['Molecule set'])
            updated_molecule_set = ast.literal_eval(current_step['Updated molecule set'])
            reaction = ast.literal_eval(current_step['Reaction'])[0]
            product = extract_molecules_from_output(current_step['Product'])
            reactants = ast.literal_eval(current_step['Reactants'])

            #step 1: check molecules' validity
            starting_signal = True
            if current_step_index == 0:
                mdd = set(molecule_set).issubset(set(target_smi))
                if not mdd:
                    starting_signal = False

            # Product in molecule set
            product_inside = False
            if product[0] in molecule_set:
                product_inside = True
            invalid_molset_mol_id = []
            invalid_updated_mol_id = []
            
            updated_set_signals = check_validity(updated_molecule_set)
            if False in updated_set_signals:
                invalid_updated_mol_id = [index for index, value in enumerate(updated_set_signals) if not value]
    
            mol_set_signals = check_validity(molecule_set)
            if False in mol_set_signals:
                invalid_molset_mol_id = [index for index, value in enumerate(mol_set_signals) if not value]

            #check purchasability
            check_availability = False
            unavailable_mol_id = []
            if i == len(route) - 1:
                avaulibities = check_purchasable(updated_molecule_set, updated_set_signals, self.inventory)
                check_availability = True
            if check_availability == True:
                if False in avaulibities:
                    unavailable_mol_id = [index for index, value in enumerate(avaulibities) if not value]
    
            #step 2
            reaction_valid, updated_set_valid, reaction_existance = False, False, False
            if ':' in reaction:
                keys = [key for key, value in self.original_template_dict.items() if value == reaction]
                if len(keys) == 1:
                    reaction_existance = True
                    reaction_key = keys[0]
                else:
                    reaction_existance = False
                    reaction_key = None
            else:
                reaction_existance, reaction_key = is_reaction_in_dict(reaction, self.template_dict)
            if reaction_key == None:
                new_reaction = reaction
            else:
                new_reaction = self.original_template_dict[reaction_key]
            

            if reaction_existance == True:
                reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants)
            
            if current_step_index == 0:
                if reaction_key == None:
                    reaction_existance, reaction_key, new_reaction = self.blurry_search(reaction, product[0])
                elif (product[0], reaction_key) in self.explored_reaction:
                    reaction_existance, reaction_key, new_reaction = self.blurry_search(reaction, product[0])
                elif reaction_existance == True and reaction_valid == False:
                    reaction_existance, reaction_key, new_reaction = self.blurry_search(reaction, product[0])  
            
                if reaction_key == None:
                    new_reaction = reaction
                else:
                    new_reaction = self.original_template_dict[reaction_key]
                    reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants)    
            if (
                len(invalid_molset_mol_id) == 0 and
                len(invalid_updated_mol_id) == 0 and
                reaction_valid and
                updated_set_valid and
                starting_signal and
                product_inside
            ):
                step_validity = True

            # Construct the dictionary
            step_info = {
                "target_smi": target_smi,
                "starting_signal": starting_signal,
                "product_inside": product_inside,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated_molecule_set,
                "reaction": new_reaction,
                "reaction_key": reaction_key,
                "product": product,
                "reactants": reactants,
                "updated_set_signals": updated_set_signals,
                "invalid_updated_mol_id": invalid_updated_mol_id,
                "mol_set_signals": mol_set_signals,
                "invalid_molset_mol_id": invalid_molset_mol_id,
                "check_availability": check_availability,
                "unavailable_mol_id": unavailable_mol_id,
                "reaction_existance": reaction_existance,
                "reaction_valid": reaction_valid,
                "updated_set_valid": updated_set_valid
            }

            # Store the tuple in the results list
            results.append((current_step_index, step_validity, step_info))
        return results

    def check_route_extra(self, target_smi, route, first_evaluation):
        """
        Check if the route is valid
        """
        results = []
        for i in range(len(route)):
            current_step_index = i
            current_step = route[current_step_index]
            step_id, is_valid, current_evaluation = first_evaluation[current_step_index]

            step_validity = False
            molecule_set = ast.literal_eval(current_step['Molecule set'])
            updated_molecule_set = ast.literal_eval(current_step['Updated molecule set'])
            reaction = ast.literal_eval(current_step['Reaction'])[0]
            product = extract_molecules_from_output(current_step['Product'])
            reactants = ast.literal_eval(current_step['Reactants'])

            #step 1: check molecules' validity
            starting_signal = True
            if current_step_index == 0:
                mmd = set(molecule_set).issubset(set(target_smi))
                if not mmd:
                    starting_signal = False

            # Product in molecule set
            product_inside = False
            if product[0] in molecule_set:
                product_inside = True

            invalid_molset_mol_id = []
            invalid_updated_mol_id = []
            
            updated_set_signals = check_validity(updated_molecule_set)
            if False in updated_set_signals:
                invalid_updated_mol_id = [index for index, value in enumerate(updated_set_signals) if not value]
    
            mol_set_signals = check_validity(molecule_set)
            if False in mol_set_signals:
                invalid_molset_mol_id = [index for index, value in enumerate(mol_set_signals) if not value]


            check_availability = False
            unavailable_mol_id = []
            if i == len(route) - 1:
                avaulibities = check_purchasable(updated_molecule_set, updated_set_signals, self.inventory)
                check_availability = True
            if check_availability == True:
                if False in avaulibities:
                    unavailable_mol_id = [index for index, value in enumerate(avaulibities) if not value]
    
            #step 2
            reaction_valid, updated_set_valid = False, False
            reaction_existance, reaction_key = current_evaluation['reaction_existance'], current_evaluation['reaction_key']
            
            new_reaction = current_evaluation["reaction"]

            if reaction_existance == True:
                reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants)
    
    
            if (
                len(invalid_molset_mol_id) == 0 and
                len(invalid_updated_mol_id) == 0 and
                reaction_valid and
                updated_set_valid and
                starting_signal and
                product_inside
            ):
                step_validity = True
            if step_validity == True:
                self.explored_reaction.add((product[0], reaction_key))
            # Construct the dictionary
            step_info = {
                "target_smi": target_smi,
                "starting_signal": starting_signal,
                "product_inside": product_inside,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated_molecule_set,
                "reaction": new_reaction,
                "reaction_key": reaction_key,
                "product": product,
                "reactants": reactants,
                "updated_set_signals": updated_set_signals,
                "invalid_updated_mol_id": invalid_updated_mol_id,
                "mol_set_signals": mol_set_signals,
                "invalid_molset_mol_id": invalid_molset_mol_id,
                "check_availability": check_availability,
                "unavailable_mol_id": unavailable_mol_id,
                "reaction_existance": reaction_existance,
                "reaction_valid": reaction_valid,
                "updated_set_valid": updated_set_valid
            }

            # Store the tuple in the results list
            results.append((current_step_index, step_validity, step_info))
        return results

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)
        self.oracle.route_buffer = {}

    @property
    def route_buffer(self):
        return self.oracle.route_buffer

    @property
    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
    def rewards(self, route_evaluation):
        return self.oracle(self.inventory, route_evaluation)
    

    def optimize(self, target, route_list, all_fps, config, seed=0, project="test"):
        self.reset()
        self.seed = seed 
        self.oracle.task_label = self.model_name + "_" + target + "_" + str(seed)
        self._optimize(target, route_list, all_fps, config)
        if self.args.log_results:
            self.log_result()
        self.save_result(self.model_name + "_" + target + "_" + str(seed))
        
    
    def query_LLM(self, question, model="gpt-4o", temperature=0.0):
        openai.api_type = 'azure'
        openai.api_base = ''
        openai.api_version = ''
        openai.api_key = ''
        message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]

        prompt1 = question
        message.append({"role": "user", "content": prompt1})

        params = {
            "engine": "gpt-4o",
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": message
        }

        for retry in range(3):
            try:
                response = openai.ChatCompletion.create(**params)["choices"][0]["message"]["content"]
                message.append({"role": "assistant", "content": response})
                break
            except Exception as e:
                print(f"{type(e).__name__} {e}")


        print("=>")
        return message, response
    

    def fix_reaction_error(self, routes, stepwise_results):
        updated_routes = []
        for i in range(len(routes)):
            step_id, is_valid, data = stepwise_results[i]
            if data['reaction_existance'] == True and data['reaction_valid'] == True:
                reaction_smiles = data['reaction']

                reactants = [Chem.MolFromSmiles(smi) for smi in data['reactants']]
                products = [Chem.MolFromSmiles(smi) for smi in data['product']]
            
                target_rd = rdchiralReactants(data['product'][0])
                reaction_outputs = run_retro(target_rd, reaction_smiles)
                if len(reaction_outputs) > 1:
                    reaction_outputs = self.rank_reactants(reaction_outputs)
                reactants_generated = [reactant for reactant in reaction_outputs[0]]
            
                reactants_smiles = set(reactants_generated)

                products_smiles = {smi for smi in data['product']}
                original_molecule_set = [Chem.MolFromSmiles(smi) for smi in data['molecule_set']]
                original_set = {Chem.MolToSmiles(mol) for mol in original_molecule_set if mol is not None}
            
                updated_mol_set = (original_set | reactants_smiles) - products_smiles
                data['Updated molecule set'] = list(updated_mol_set)
                routes[i]['Reactants'] = str(list(reactants_smiles))
                routes[i]['Updated molecule set'] = str(data['Updated molecule set'])

        return routes
    def rank_reactants(self, reactants_list):
        """
        Rank reactants based on the number of products generated
        """
        scores = [self.oracle.reward(self.inventory, reactant) for reactant in reactants_list]
        sorted_list = [x for _, x in sorted(zip(scores, reactants_list), key=lambda pair: pair[0], reverse=True)]
        return sorted_list
def check_and_update_routes(routes, target_list):
    import ast

    routes[0]['Molecule set'] = str(target_list)
    for i in range(1, len(routes)):
        current_updated_set = ast.literal_eval(routes[i]['Molecule set'])
        previous_molecule_set = ast.literal_eval(routes[i - 1]['Updated molecule set'])


        if set(current_updated_set) != set(previous_molecule_set):
            print(f"Mismatch found at step {i}:")
            print(f"  Previous Molecule set: {previous_molecule_set}")
            print(f"  Current Updated molecule set: {current_updated_set}")

            routes[i]['Molecule set'] = str(previous_molecule_set)
            print(f"  Updated step {i} to match previous Molecule set.")

    print("\nAll steps checked and updated where necessary.")
    return routes



def map_reaction(routes, stepwise_results):
    updated_routes = []
    for i in range(len(routes)):
        step_id, is_valid, data = stepwise_results[i]
        if data['reaction_existance'] == True:
            reaction_smiles = data['reaction']

            routes[i]['Reaction'] = str([reaction_smiles])

    return routes

def extract_molecules_from_output(output):
    try:

        parsed_output = ast.literal_eval(output)

        if isinstance(parsed_output, list):
            return parsed_output
        elif isinstance(parsed_output, str):
            return [parsed_output]
        else:
            return []
    except (ValueError, SyntaxError):

        return []