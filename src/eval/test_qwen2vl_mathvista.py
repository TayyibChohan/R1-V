from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from datasets import load_dataset, Image as HFImage



PROMPT_PATH=None
MODEL_PATH="/data/tayyibc/r1-v_out/checkpoint-700/"
BSZ=64 # reduce it if GPU OOM
os.makedirs("./logs", exist_ok=True)
OUTPUT_PATH=f"./logs/mathvista_testmini_eval{MODEL_PATH.split('/')[-1]}_bsize{BSZ}.json"
DATASET= "AI4Math/MathVista"

use_hf_data = True

QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."


def extract_number_answer(output_str):
    # Match numbers, words, or single characters in answer tags
    answer_pattern = r'<answer>\s*([^<]+)\s*</answer>'
    match = re.search(answer_pattern, output_str)
    
    if match:
        answer_text = match.group(1).strip()
        # Try to convert to integer if possible
        try:
            return int(answer_text)
        except ValueError:
            # Check if it's a float
            try:
                return float(answer_text)
            except ValueError:
                # Return string otherwise
                return answer_text
    return None




def main():
    # load the model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # load the dataset
    print("Loading dataset...")
    data = []
    messages = []
    if use_hf_data:
        dataset = load_dataset(DATASET, split="testmini")
        #use lambda expressions to relevent fields
        data = dataset.map(lambda x: {
            "image": x["decoded_image"] if (x["decoded_image"].width > 28 and x["decoded_image"].height > 28) else None,
            "question": x["question"] if x["choices"] is None else x["question"] + " Choose from the following options " + " ".join(x["choices"]),
            "ground_truth": x["answer"]
        }, remove_columns=dataset.column_names)
        #remove None values
        data = data.cast_column("image", HFImage())
        data = data.filter(lambda x: x["image"] is not None)
        # Convert Dataset to list of dictionaries correctly
        data = [{"image": example["image"], 
                 "question": example["question"], 
                 "ground_truth": example["ground_truth"]} 
                for example in data]
    else:
        with open(PROMPT_PATH, "r") as f:
            for line in f:
                data.append(json.loads(line))

    # process the data
    for i in data:
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": i['image'],
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=i['question'])
                }
            ]
        }]
        messages.append(message)

    all_outputs = []  # List to store all answers
    correct_number = 0
    final_output = []  # List to store final output

    # Modify the process_vision_info handling
    print("Processing data in batches...")
    for i in tqdm(range(0, len(messages), BSZ)):
        batch_messages = messages[i:i + BSZ]
        
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        all_outputs.extend(batch_output_text)
        print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")
   
    # extract the answers
    print("Extracting answers...")
    for input_example, model_output in zip(data,all_outputs):
        original_output = model_output
        ground_truth = input_example['ground_truth']
        model_answer = extract_number_answer(original_output)
        
        # Create a result dictionary for this example
        result = {
            'question_text': input_example['question'],
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer': model_answer
        }
        final_output.append(result)
        
        # Count correct answers
        if model_answer is not None and model_answer == ground_truth:
            correct_number += 1

    # Calculate and print accuracy
    accuracy = correct_number / len(data) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
    # Save results to a JSON file
    output_path = OUTPUT_PATH
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2)
    print(f"Results saved to {output_path}")
    

if __name__ == "__main__":
    main()



