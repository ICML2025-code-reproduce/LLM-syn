from __future__ import print_function

import random
from typing import List

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
from rdkit import DataStructs
import ast
from syntheseus import Molecule
import rdkit.Chem.AllChem as AllChem
rdBase.DisableLog('rdApp.error')
from utils import *
from optimizer import BaseOptimizer, check_and_update_routes, map_reaction, extract_molecules_from_output
import os
import openai
import re
import copy
from concurrent.futures import ThreadPoolExecutor
from route import *
from modification_prompt import modification_hints, construct_modification_prompt
from utils import *
import time


MINIMUM = 1e-10





def make_mating_pool(population_mol: List, population_scores, offspring_size: int):

    # scores -> probs 
    population_scores = [-(s - MINIMUM) for s in population_scores]

    weights = np.exp(-np.array(population_scores))  # Exponentially invert scores
    probabilities = weights / weights.sum()  # Normalize to get probabilities
    sampled_index = np.random.choice(len(population_mol), p=probabilities, size=offspring_size, replace=True)
    mating_pool = [population_mol[i] for i in sampled_index]
    return mating_pool




class planning_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "planning"

    def exploration(self, target_smi, rag_tuples):
        population = []
        route_list = [t[1] for t in rag_tuples]
        sims_list = [t[0] for t in rag_tuples]


        sum_scores = sum(sims_list)
        population_probs = [p / sum_scores for p in sims_list]
        while True:
            try:
                sampled_index = np.random.choice(len(route_list), p=population_probs, size=3, replace=False)
                sampled_routes = [route_list[i] for i in sampled_index]
                examples = ''
                for i in sampled_routes:
                    examples = examples + '<ROUTE>\n'+ str(process_reaction_routes(i)) + '\n</ROUTE>\n'
                examples = examples + '\n'
                initial_prompt = '''
                As a professional chemist specialized in synthesis analysis, you are tasked with generating a full retrosynthesis route for a target molecule provided in SMILES format.
                A retrosynthesis route is a series of retrosynthesis steps that starts from the target molecule and ends with some commercially purchasable compounds. The reactions are from the USPTO dataset. Please also take reactions in stereochemistry into consideration.
                '''

                task_description = '''
                My target molecule is: {}

                To assist you, example retrosynthesis routes that are either close to the target molecule or representative will be provided.\n {}

                Please propose a retrosynthesis route for my target molecule. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge.
                '''.format(target_smi,examples)

                requirements = '''
                The route should be a list of steps wrapped in <ROUTE></ROUTE> with <EXPLAINATION></EXPLAINATION> after it. Each step in the list should be a dictionary. You need to keep a molecule set in which are the molecules we need to synthesize or purchase. In each step, you need to select a molecule from the 'Molecule set'
                as the prodcut molecule in this step and use a backward reaction to find the reactants. After taking the backward reaction in this step, you need to remove the product molecule from the molecule set and add the reactants you find into the molecule set and then name
                this updated set as the 'Updated molecule set' in this step. In the next step, the starting molecule set should be the 'Updated molecule set' from the previous step. In the last step, all the molecules in the 'Updated molecule set' should be purchasable. Here is an example:
                corresponds to a set of molecules that are commercially available. Here is an example:

                <ROUTE>
                [   
                    {
                        'Molecule set': "[Target Molecule]",
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
                Requirements: 1. The 'Molecule set' contains molecules we need to synthesize at this stage. In the first step, it should be the target molecule. In the following steps, it should be the 'Updated molecule set' from the previous step.\n
                2. The 'Rational' part in each step should be your analysis for syhthesis planning in this step. It should be in the string format wrapped with \'\'\n
                3. 'Product' is the molecule we plan to synthesize in this step. It should be from the 'Molecule set'. The molecule should be a molecule from the 'Molecule set' in a list. The molecule smiles should be wrapped with \'\'.\n
                4. 'Reaction' is a reaction which can synthesize the product molecule. It should be in a list. The reaction template should be in SMILES format. For example, [Product>>Reactant1.Reactant2].\n
                5. 'Reactants' are the reactants of the reaction. It should be in a list. The molecule smiles should be wrapped with \'\'.\n
                6. The 'Updated molecule set' should be molecules we need to purchase or synthesize after taking this reaction. To get the 'Updated molecule set', you need to remove the product molecule from the 'Molecule set' and then add the reactants in this step into it. In the last step, all the molecules in the 'Updated molecule set' should be purchasable.\n
                7. In the <EXPLAINATION>, you should analyze the whole route and ensure the molecules in the 'Updated molecule set' in the last step are all purchasable.\n'''
                question = initial_prompt + requirements + task_description
                message, answer = self.query_LLM(question, temperature=0.7)
                # Converting the extracted string to a Python list of dictionaries
                print(answer)
                match = re.search(r'<ROUTE>(.*?)<ROUTE>', answer, re.DOTALL)
                if match == None:
                    match = re.search(r'<ROUTE>(.*?)</ROUTE>', answer, re.DOTALL)

                route_content = match.group(1)

                route = ast.literal_eval(route_content)
                comp1 = ast.literal_eval(route[-1]['Updated molecule set'])
                comp2 = ast.literal_eval(route[-2]['Updated molecule set'])
                last_step_reactants = route[-1]['Reactants']
                if set(comp1) == set(comp2) or last_step_reactants == "" or last_step_reactants == "[]" or last_step_reactants == "None" or last_step_reactants == "[None]":
                    route = route[:-1]
                    print('Route cleaned!')
                for step in route:
                    temp = ast.literal_eval(step['Molecule set'])
                    temp = ast.literal_eval(step['Reaction'])[0]
                    temp = extract_molecules_from_output(step['Product'])[0]
                    temp = ast.literal_eval(step['Reactants'])[0]
                    temp = ast.literal_eval(step['Updated molecule set'])

                break
            except Exception as e:
                print(f"Error in generating the initial population: {e}")
                continue
        route_class_item = Route(target_smi)
        checked_route, final_evaluation = self.sanitize([target_smi], route)
        score = self.rewards(final_evaluation)
        route_class_item.add_route(checked_route, final_evaluation)
        route_class_item.update_reward(score)
        route_class_item.update_evaluation(final_evaluation)

        return route_class_item

    def modification(self, combined_list, population_routes, all_fps, route_list, inventory):
        parent_a = random.choice(combined_list)
        sampled_route = parent_a[1]
        count = 0
        while True:
            try:
                count = count + 1
                if count >= 10:
                    count = 0
                    parent_a = random.choice(combined_list)
                    sampled_route = parent_a[1]
                route = sampled_route.validated_route
                new_route_item = copy.deepcopy(sampled_route)
                evaluation = new_route_item.evaluation
                molecule_list = route[-1]['Updated molecule set']
                unpurchasable_list = check_availability(molecule_list, inventory)
                
                retrieved_routes = modification_hints(unpurchasable_list, all_fps, route_list)

                new_q = construct_modification_prompt(unpurchasable_list, retrieved_routes, evaluation, inventory)
                new_m, new_a = self.query_LLM(new_q, temperature=0.7)

                match = re.search(r'<ROUTE>(.*?)<ROUTE>', new_a, re.DOTALL)
                if match == None:
                    match = re.search(r'<ROUTE>(.*?)</ROUTE>', new_a, re.DOTALL)
                    #print(answer)
                print(new_a)
                route_content = match.group(1)

                new_route = ast.literal_eval(route_content)
                comp1 = ast.literal_eval(new_route[-1]['Updated molecule set'])
                comp2 = ast.literal_eval(new_route[-2]['Updated molecule set'])
                last_step_reactants = new_route[-1]['Reactants']
                if set(comp1) == set(comp2) or last_step_reactants == "" or last_step_reactants == "[]" or last_step_reactants == "None" or last_step_reactants == "[None]":
                    new_route = new_route[:-1]
                    print('Route cleaned!')
                for step in new_route:
                    temp = ast.literal_eval(step['Molecule set'])
                    temp = ast.literal_eval(step['Reaction'])[0]
                    temp = extract_molecules_from_output(step['Product'])[0]
                    temp = ast.literal_eval(step['Reactants'])[0]
                    temp = ast.literal_eval(step['Updated molecule set'])

                checked_route, final_evaluation = self.sanitize(molecule_list, new_route)
                if final_evaluation[0][2]['reaction_existance'] == False:
                    continue
                new_route_item.update_route(checked_route, final_evaluation)
                if not check_distinct_route(population_routes, new_route_item):
                    continue
                score = self.rewards(final_evaluation)
                
                new_route_item.update_reward(score)
                new_route_item.update_evaluation(final_evaluation)

                break
            except Exception as e:
                print(f"Error in generating the modification population {e}")

                continue

        return new_route_item

    def _optimize(self, target, route_list, all_fps, config):

        population_class = []
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
        similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
        fp = getfp(target)
    
        sims = similarity_metric(fp, [fp_ for fp_ in all_fps])

        rag_tuples = list(zip(sims, route_list))
        rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:50]
        assert len(self.oracle.route_buffer) == 0, f"route_buffer not empty: {self.oracle.route_buffer}"

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.exploration, target, rag_tuples) for _ in range(config["population_size"])]
            starting_population = [future.result() for future in futures]

        # initial population
        population_routes = starting_population

        population_scores = [route_class_item.get_reward() for route_class_item in population_routes]

        combined_list = list(zip(population_scores, population_routes))
        all_routes = copy.deepcopy(population_routes)
        while True:

            if len(self.oracle) > 10:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.route_buffer.items())[:10]])
            else:
                old_score = 0
            

            if len(self.oracle) > 5:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.route_buffer.items())[:10]])
                if 0 in population_scores:

                    self.log_intermediate(finish=True)
                    zero_count = population_scores.count(0)
                    file_path = 'stat/results.txt'
                    with open(file_path, 'a') as file:
                        file.write(f"{zero_count}\n")     
                    print('convergence criteria met, abort ...... ')
                    for route in population_routes:
                        reward = route.get_reward()
                        if reward == 0:
                            route.save_result(target)
                            
                    break

                old_score = new_score

            # new_population
            mating_pool = make_mating_pool(combined_list, population_scores, config["population_size"])

            #modification
            with ThreadPoolExecutor() as executor:

                futures = [executor.submit(self.modification, mating_pool, all_routes, all_fps, route_list, self.inventory) for _ in range(config["offspring_size"])]
                offspring_routes = [future.result() for future in futures]
            # add new_population
            population_routes += offspring_routes
            all_routes = all_routes + offspring_routes
            # stats
            old_scores = population_scores


            population_scores = [route_class_item.get_reward() for route_class_item in population_routes]
            combined_list = list(zip(population_scores, population_routes))
            combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_routes = [t[1] for t in combined_list]
            population_scores = [t[0] for t in combined_list]



            ### early stopping

            if len(self.oracle) > 5:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.route_buffer.items())[:10]])
                if 0 in population_scores:
                    self.log_intermediate(finish=True)
                    zero_count = population_scores.count(0)
                    file_path = 'stat/results.txt'
                    with open(file_path, 'a') as file:
                        file.write(f"{zero_count}\n")     
                    print('Find route, abort ...... ')
                    for route in population_routes:
                        reward = route.get_reward()
                        if reward == 0:
                            route.save_result(target)
                            
                    break

                old_score = new_score
  
            if self.finish:
                zero_count = population_scores.count(0)
                file_path = 'stat/results.txt'
                with open(file_path, 'a') as file:
                    file.write(f"{zero_count}\n")     
                print('convergence criteria met, abort ...... ')              
                break