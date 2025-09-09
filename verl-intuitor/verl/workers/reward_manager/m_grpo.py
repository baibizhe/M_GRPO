# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, Counter
import numpy as np 
import torch
from typing import Dict, Tuple, List, Optional
from verl import DataProto
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl import DataProto
from verl.utils.reward_score import (
    _default_compute_score,
    _extract_verifiable_part_of_solution,
)
import torch
from collections import defaultdict, Counter


GROUND_TRUTH_FOR_PROMPTS_THAT_NEED_TO_BE_SELF_LABELLED = "LABEL_BY_SELF_CONSISTENCY"

def majority_vote(ans_list, empty_value=''):
    ans_list = [a for a in ans_list if a is not None and str(a).strip() != '']
    if not ans_list:
        return empty_value  # 或 return 0, 或 return '[NO_ANSWER]'
    return Counter(ans_list).most_common(1)[0][0]



class M_grpo_RewardManager:
    """The reward manager.
    """
    def __init__(self, 
        tokenizer, 
        num_examine, 
        self_consistency_threshold=0.5,
        soft_reward=False,
        remove_kl_loss_from_unlabeled_examples=True,
        compute_score=None, 
        reward_fn_key='data_source',
        oversampling_keep_fraction=1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        # Parameters specific to self labeling
        self.self_consistency_threshold = self_consistency_threshold
        self.soft_reward = soft_reward
        self.remove_kl_loss_from_unlabeled_examples = remove_kl_loss_from_unlabeled_examples
        self.oversampling_keep_fraction = oversampling_keep_fraction

        assert (
            isinstance(self.oversampling_keep_fraction, float)
            and self.oversampling_keep_fraction > 0
            and self.oversampling_keep_fraction <= 1.0
        )

        # Print stuff
        print("\nSelf consistency threshold: ", self.self_consistency_threshold)
        print("Using soft reward: ", self.soft_reward)
        print("Remove KL loss from unlabelled examples: ", self.remove_kl_loss_from_unlabeled_examples)
        print(f"Keeping {self.oversampling_keep_fraction} fraction of prompts. \n")

    def get_prompt_and_response_and_ground_truth(
        self,
        data_item,
    ) -> Tuple[str, str, str, str]:
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

        solution_hidden_during_training = data_item.non_tensor_batch['reward_model'].get(
            'solution_hidden_during_training',
            ground_truth,
        )

        assert isinstance(ground_truth, str)

        return prompt_str, response_str, ground_truth, solution_hidden_during_training
    
    def calculate_fraction_of_data_at_pass_rate_interval(
        self,
        all_pass_rates: List[float],
        pass_rate_interval: Tuple[int, int],
    ) -> float:
        num_pass_rates_between_interval = 0.0
        low = pass_rate_interval[0]
        high = pass_rate_interval[1]

        for pass_rate in all_pass_rates:
            if pass_rate >= low and pass_rate <= high:
                num_pass_rates_between_interval += 1.0

        return num_pass_rates_between_interval / len(all_pass_rates)
    
    def calculate_pass_rate_distribution(
        self,
        data: DataProto,
    ) -> Dict:
        prompt_to_correctness = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]
            (
                prompt_str, 
                response_str, 
                _, 
                hidden_solution,
            ) = self.get_prompt_and_response_and_ground_truth(
                data_item=data_item,
            )

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=hidden_solution,
                extra_info=extra_info,
            )

            prompt_to_correctness[prompt_str].append(score)

        prompt_to_pass_rate = {}
        for prompt in prompt_to_correctness:
            prompt_to_pass_rate[prompt] = np.mean(prompt_to_correctness[prompt])

        all_pass_rates = []
        for prompt in prompt_to_pass_rate:
            all_pass_rates.append(prompt_to_pass_rate[prompt])

        pass_rate_intervals = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        pass_rate_metrics = {}
        for pass_rate_interval in pass_rate_intervals:
            pass_rate_metrics[f"train_pass_rates/between_{pass_rate_interval}"] = (
                self.calculate_fraction_of_data_at_pass_rate_interval(
                    all_pass_rates=all_pass_rates,
                    pass_rate_interval=pass_rate_interval,
                )
            )

        return pass_rate_metrics

    def calculate_ground_truth_answers(
        self,
        data: DataProto,
    ) -> Tuple[Dict, Dict, Dict, Optional[List]]:
        # First, make a map, {prompt x: [z: z is a generation for prompt x]}
        # and a map, {prompt x: given ground truth y}
        prompt_to_generation_map = defaultdict(list)
        prompt_to_ground_truth_map = {}
        prompt_to_hidden_solution_map = {}

        for i in range(len(data)):
            data_item = data[i]
            (
                prompt_str, 
                response_str, 
                ground_truth, 
                hidden_solution,
            ) = self.get_prompt_and_response_and_ground_truth(
                data_item=data_item,
            )
            # if i ==10:
            #     print(f'response_str: {response_str}')
            # 格式化打印
            # print(
            #     f"\n===== Debug Info =====\n"
            #     f"response_str:\n{response_str}\n\n"
            #     f"ground_truth:\n{ground_truth}\n\n"
            #     f"======================\n"
            # )
            extracted_answer = _extract_verifiable_part_of_solution(
                data_source=data_item.non_tensor_batch[self.reward_fn_key],
                solution_str=response_str,
            )
            # print(
            #     f"\n===== Debug Info =====\n"
            #     f"extracted_answer:\n{extracted_answer}\n\n"
            #     f"======================\n"
            # )

            if extracted_answer is not None:
                prompt_to_generation_map[prompt_str].append(extracted_answer)

            prompt_to_ground_truth_map[prompt_str] = ground_truth
            prompt_to_hidden_solution_map[prompt_str] = hidden_solution

        # Next, for prompts for which the gold solution needs to be calculated by self-consistency
        # we generate them, otherwise obtain it from gold label

        # keep track of prompts that are labeled online
        num_all_prompts = 0.0
        num_prompts_labeled_via_self_consistency = 0.0
        num_prompts_correctly_labeled = 0.0

        prompt_to_answer_reward_map = {}

        # NOTE: this is only used when self.soft_reward is False
        prompt_to_majority_answer_fraction = {}

        for prompt_str in prompt_to_ground_truth_map:
            num_all_prompts += 1
            # print(f"\n===== Debug Info =====\n")
            # print(f'prompt_to_ground_truth_map:{prompt_to_ground_truth_map}')
            # print(f'prompt_to_generation_map:{prompt_to_generation_map}')
            
            # Gold reward available for this datapoint
            # So we do not need to calculate self labeling reward
            if False:
                pass
            # No Gold reward available for this datapoint
            # So NEED TO SELF-LABEL
            else:
               
                all_solutions = prompt_to_generation_map[prompt_str]
                # print(f"\n===== Debug Info =====\n")
                # print(f'all_solutions: {all_solutions}') 
                # print(f'prompt_str: {prompt_str}') 
                # Case 1: At least one parsable answer exists
                if len(all_solutions) > 0:
                    counts = Counter(all_solutions)

                    # Calculate soft reward label
                    if self.soft_reward:
                        sorted_by_frequency = counts.most_common()
                        answer_to_reward_map = {}

                        for index in range(len(sorted_by_frequency)):
                            curr_answer, answer_frequency = sorted_by_frequency[index]
                            answer_reward = float(answer_frequency) / len(all_solutions)
                            answer_to_reward_map[curr_answer] = answer_reward

                        prompt_to_answer_reward_map[prompt_str] = answer_to_reward_map

                    # calculate self consistency level
                    else:

                        majority_answer, majority_frequency = counts.most_common(1)[0]
                        majority_answer_appears_this_fraction_of_the_time = (
                            float(majority_frequency) / len(all_solutions)
                        )
                        # print(f"\n===== Debug Info =====\n")
                        # print(f'counts: {counts}====')
                        # print(f'majority_answer: {majority_frequency}====')
                        # print(f'majority_frequency: {majority_answer}====')
                        
                        if (
                            majority_answer_appears_this_fraction_of_the_time >= self.self_consistency_threshold
                        ):
                            prompt_to_answer_reward_map[prompt_str] = {
                                majority_answer: 1.0,
                            }

                            # would be used to potentially throw out data
                            prompt_to_majority_answer_fraction[prompt_str] = majority_answer_appears_this_fraction_of_the_time

                            num_prompts_labeled_via_self_consistency += 1

                            hidden_answer = prompt_to_hidden_solution_map[prompt_str]
                            if majority_answer == hidden_answer:
                                num_prompts_correctly_labeled += 1

                        # None of the generations crossed self-consistency threshold
                        # So we don't label anything
                        else:
                            prompt_to_answer_reward_map[prompt_str] = None

                # Case 2: No parsable answer exists
                # No self labeling
                else:
                    prompt_to_answer_reward_map[prompt_str] = None

        
        if num_all_prompts != 0:
            fraction_labeled = num_prompts_labeled_via_self_consistency / num_all_prompts
        else:
            fraction_labeled = 0.0

        if num_prompts_labeled_via_self_consistency != 0:
            fraction_correctly_labeled = num_prompts_correctly_labeled / num_prompts_labeled_via_self_consistency
        else:
            fraction_correctly_labeled = 0.0

        self_labeling_metrics = {
            "self_labeling_metrics/fraction_labeled_via_self_consistency": fraction_labeled,
            "self_labeling_metrics/fraction_correctly_labeled": fraction_correctly_labeled,
        }
        
        # Calculate prompts that we will throw out
        prompts_to_keep = None
        if (
            not self.soft_reward 
            and self.oversampling_keep_fraction < 1.0
        ):
            prompts_to_keep = []

            prompt_majority_fraction_pairs = []
            for _prompt in prompt_to_ground_truth_map:
                # Prompts with gold labels, no need to throw them out
                if prompt_to_ground_truth_map[_prompt] != GROUND_TRUTH_FOR_PROMPTS_THAT_NEED_TO_BE_SELF_LABELLED:
                    prompts_to_keep.append(_prompt)

                else:
                    answer_reward_map = prompt_to_answer_reward_map[_prompt]
                    if answer_reward_map is not None:
                        assert _prompt in prompt_to_majority_answer_fraction
                        majority_fraction = prompt_to_majority_answer_fraction[_prompt]

                        prompt_majority_fraction_pairs.append((_prompt, majority_fraction))

            prompt_majority_fraction_pairs = sorted(
                prompt_majority_fraction_pairs,
                key=lambda x: x[1],
                reverse=True,
            )

            prompts_in_sorted_order = [
                prompt_majority_fraction_tuple[0]
                for prompt_majority_fraction_tuple in prompt_majority_fraction_pairs
            ]

            num_total_prompts = len(prompt_to_ground_truth_map)
            num_prompts_to_keep = int(num_total_prompts * self.oversampling_keep_fraction)
            assert num_prompts_to_keep >= 1

            prompts_to_keep = prompts_to_keep + prompts_in_sorted_order
            prompts_to_keep = prompts_to_keep[:num_prompts_to_keep]

        return (
            self_labeling_metrics, 
            prompt_to_ground_truth_map, 
            prompt_to_answer_reward_map,
            prompts_to_keep,
        )

    def get_modal_accuracy_per_dataset(self, prompt_list, data_source_list, gt_list, extracted_answer_list) -> defaultdict: 
        """
        This is a modified version of the the preprocess_plotting_matrices function.
        Args:
            prompt_list: list of prompts
            data_source_list: list of data sources
            gt_list: list of ground truths
            extracted_answer_list: list of extracted answers
        Returns:
            modal_accuracy_dict : defaultdict of modal accuracy scores

        NOTE: currently, we take the mode over the *extracted answers*, and then calculate the accuracy based on the *extracted* answer (and not the raw llm output).
        This can be changed, but it is the only thing implemented for now.
        """


        print(len(prompt_list))
        N = len(set(prompt_list))
        n = len(prompt_list) // N

        assert n * N == len(prompt_list)

        prediction_matrix = np.array(extracted_answer_list).reshape(N, n)
        ground_truths_matrix = np.array(gt_list).reshape(N, n)
        data_source_matrix = np.array(data_source_list).reshape(N, n)

        mode_accuracy = defaultdict(list)

        for i in range(N): # for each prompt
            ith_pred_ = prediction_matrix[i]
            ith_pred = [x for x in ith_pred_ if x != ""]
            ith_pred = [x for x in ith_pred if x is not None]
            gt = ground_truths_matrix[i][0]
            data_source = data_source_matrix[i][0]
            
            if len(ith_pred) == 0:
                mode = None
            else:
                counter = Counter(ith_pred)
                mode, _ = counter.most_common(1)[0]
                try:
                    mode = "\\boxed{"+mode+"}"
                except:
                    mode = None
            # _list.append(self.compute_score(solution_str=mode, ground_truth=gt, data_source=data_source))
            mode_accuracy[data_source].append(self.compute_score(solution_str=mode, ground_truth=gt, data_source=data_source))

        return mode_accuracy

    def __call__(self, data: DataProto, return_dict=False, log_threshold_plot=False):
        """We will expand this function gradually based on the available datasets.
        Threshold plotting is not implemented here.
        """
        

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        kl_loss_mask = torch.ones_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # First, we calculate labels using self consistency
        (
            self_labeling_metrics, 
            prompt_to_ground_truth_map,
            prompt_to_answer_reward_map,
            prompts_to_keep,
        ) = self.calculate_ground_truth_answers(
            data=data,
        )

        num_prompts_self_labeled = 0.0
        num_correct_among_self_labeled_prompts = 0.0
        num_train_generations_correct = 0.0 
        num_total_generations_so_far = 0.0
        
        gt_list, prompt_list, data_source_list, extracted_answer_list, llm_response_list = [], [], [], [], []
        indices_to_keep = []

        for i in range(len(data)):
            num_total_generations_so_far += 1
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # We comment this out because we are doing self-learning!
            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # Use the calculated ground truth instead
            ground_truth = prompt_to_ground_truth_map[prompt_str]
            solution_hidden_during_training = data_item.non_tensor_batch['reward_model'].get(
                'solution_hidden_during_training',
                ground_truth,
            )

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            num_train_generations_correct += self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=solution_hidden_during_training,
                extra_info=extra_info,
            )

            # Prompts that need to be self-labeled
            if ground_truth == GROUND_TRUTH_FOR_PROMPTS_THAT_NEED_TO_BE_SELF_LABELLED:
                answer_reward_map = prompt_to_answer_reward_map[prompt_str] 

                # Prompts that remains unlabeled
                # We remove rewards from it
                if answer_reward_map is None:
                    score = 0.0
                    
                    if self.remove_kl_loss_from_unlabeled_examples:
                        kl_loss_mask[i, :] = 0.0

                else:
                    score = 0.0

                    for curr_answer in answer_reward_map:
                        partial_score = self.compute_score(
                            data_source=data_source,
                            solution_str=response_str,
                            ground_truth=curr_answer,
                            extra_info=extra_info,
                        )

                        score += partial_score * answer_reward_map[curr_answer]

                    # Consistency threshold/majority voting label
                    if not self.soft_reward:
                        num_prompts_self_labeled += 1

                        # score calculated using hidden, unavailable labels
                        # used for logging/tracking purposes
                        num_correct_among_self_labeled_prompts += self.compute_score(
                            data_source=data_source,
                            solution_str=response_str,
                            ground_truth=solution_hidden_during_training,
                            extra_info=extra_info,
                        )


            # Prompts that have ground_truth labels available
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

             # Code for mode accuracy
            extracted_answer = _extract_verifiable_part_of_solution(
                data_source=data_source,
                solution_str=response_str,
            )
            gt_list.append(ground_truth)
            prompt_list.append(prompt_str)
            data_source_list.append(data_source)
            extracted_answer_list.append(extracted_answer)
            llm_response_list.append(response_str)
            
            if (
                prompts_to_keep is not None
                and prompt_str in prompts_to_keep
            ):
                indices_to_keep.append(i)

        if num_prompts_self_labeled != 0:
            average_accuracy_on_self_labeled_prompts = (
                float(num_correct_among_self_labeled_prompts) / num_prompts_self_labeled
            )
        else:
            average_accuracy_on_self_labeled_prompts = 0.0

        ground_truth_accuracy_on_training_generation_batch = (
            num_train_generations_correct / num_total_generations_so_far
        )

        if return_dict:
            self_labeling_metrics["self_labeling_metrics/average_accuracy_on_self_labeled_prompts"] = (
                average_accuracy_on_self_labeled_prompts
            )
            self_labeling_metrics["training_batch/ground_truth_accuracy_on_training_set"] = (
                ground_truth_accuracy_on_training_generation_batch
            )

            pass_rate_distribution = self.calculate_pass_rate_distribution(
                data=data,
            )

            modal_accuracy_dict = self.get_modal_accuracy_per_dataset(
                prompt_list=prompt_list,
                data_source_list=data_source_list,
                gt_list=gt_list,
                extracted_answer_list=extracted_answer_list,
            )
            
            if prompts_to_keep is None:
                indices_to_keep = None

            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "kl_loss_mask": kl_loss_mask,
                "self_labeling_metrics": self_labeling_metrics,
                "pass_rate_distribution": pass_rate_distribution,
                "modal_accuracy_dict": modal_accuracy_dict,
                "indices_to_keep": indices_to_keep,
            }
        
        else:
            return reward_tensor