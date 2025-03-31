# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from PIL import Image
import io
from datasets import Image as HFImage
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # print("solution:", solution)
    for idx, (content, sol) in enumerate(zip(contents, solution)):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                # print("sol:", sol)
                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                if ground_truth == "":
                    ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
                # Log image URL if available in kwargs
                if "image_url" in kwargs and idx < len(kwargs["image_url"]):
                    f.write(f"Image URL: {kwargs['image_url'][idx]}\n")
                f.write("-" * 50 + "\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    print("Using dataset: ", script_args.dataset_name)
    if script_args.dataset_name == "AI4Math/MathVista":
        script_args.dataset_train_split = "testmini"
        script_args.dataset_test_split = "no"

        #keep only the required columns for the dataset and rename them only include if the answer_type not "free_form"
        print("Modifying the dataset")
        # dataset = dataset.filter(lambda x: x["answer_type"] != "free_form")
        # print("Sample answer:", sample["answer"])
        # use_image = True
        dataset = dataset.map(
            lambda x: {
                "problem": x["question"] if x["choices"] is None else x["question"] + " Choose from the following options " + " ".join(x["choices"]),
                "solution": x["answer"],
                #use the decoded_image if use_image is true, otherwise use create a dummy blank image
                # "image": x["decoded_image"] if use_image else Image.new("RGB", (256, 256), (255, 255, 255)),
                #use the decoded_image for the image if the dimensions are valid, otherwise skip the image
                "image": x["decoded_image"] if (x["decoded_image"].width > 28 and x["decoded_image"].height > 28) else None,
                "image_url": x["image"] # for debug
            }
        )
        dataset = dataset.cast_column("image", HFImage())
        dataset = dataset.filter(lambda x: x["image"] is not None)
        
        print("Dataset modified")

        required_columns = ["problem", "solution", "image_url", "image"]
        for split in dataset.keys():
            for column in dataset[split].column_names:
                if column not in required_columns:
                    dataset[split] = dataset[split].remove_columns(column)

        print("Dataset splits:", list(dataset.keys()))
        print("Columns in test split:", dataset["test"].column_names)
        print("Dataset splits:", list(dataset.keys()))
        print("Dataset features (test split):", dataset["test"].column_names)
        print("Dataset Size (test split):", len(dataset["test"]))
        print("Dataset Size (testmini split):", len(dataset["testmini"]))
        sample = dataset[script_args.dataset_train_split][0]
        print("Sample problem:", sample["problem"])
        # print(type(sample["image"])) 
        # if isinstance(sample["image"], dict):
        #     print("image is dict")
        #     print(sample["image"].keys())
        print("solution:", sample["solution"])


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages") if "messages" in dataset[script_args.dataset_train_split].features else dataset

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    #setup cosine learning rate scheduler
    # if training_args.learning_rate is not None:
    #     lr_scheduler = get_cosine_schedule_with_warmup(
    #         optimizer=optimizer,
    #         num_warmup_steps=training_args.warmup_steps,
    #         num_training_steps=training_args.max_steps,
    #     )
    # else:
    #     lr_scheduler = None
    #     optimizer = None

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        # optimizers=(optimizer, lr_scheduler),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
