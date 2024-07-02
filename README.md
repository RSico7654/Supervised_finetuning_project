# Fine-Tuning Llama-2 with Tiny Supervised Dataset

## Project Overview
This project was developed as part of the Intel Unnati Industrial Training program. The goal was to fine-tune the "NousResearch/Llama-2-7b-chat-hf" language model using the "llamafactory/tiny-supervised-dataset" dataset. Through this project, we gained practical experience in model training and optimization, specifically within constrained computational environments.

## Model and Dataset
- **Base Model:** NousResearch/Llama-2-7b-chat-hf (https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)
- **Fine-Tuning Dataset:** llamafactory/tiny-supervised-dataset (https://huggingface.co/datasets/llamafactory/tiny-supervised-dataset)

## Project Objectives
- Fine-tune the Llama-2 model to improve its performance on the specified dataset.
- Optimize the training process to handle memory constraints and runtime errors.
- Gain hands-on experience with advanced AI models and their deployment.

## Key Learnings
- **Model Training:** Learned the intricacies of training a large language model, including data preprocessing, setting training parameters, and handling large datasets.
- **Optimization:** Addressed challenges related to memory usage and runtime errors, enhancing the model's efficiency.
- **Problem-Solving:** Developed problem-solving and adaptability skills in the context of machine learning projects.

## Technical Details
  **Custom Environment and Kernel Preparation**
    *Terminal Commands*
-      1. conda create -n itrex python=3.10 -y (creating a conda environment)
-      2. conda activate itrex                 (activating the environment)
-      3. pip install intel-extension-for-transformers
-      4. git clone https://github.com/intel/intel-extension-for-transformers.git 
-      5. cd ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/
-      6. pip install -r requirements_cpu.txt
-      7. pip install -r requirements.txt
-      8. huggingface-cli login
-      9. python3 -m pip install jupyter ipykernel 
-      10. python3 -m ipykernel install --name neural-chat-3 --user
-      11. pip install torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate einops 
-      12. pip install datasets

- **Tokenizer and Model Preparation:**
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
  from peft import LoraConfig
  from trl import SFTTrainer
  import torch

  base_model = "NousResearch/Llama-2-7b-chat-hf"
  new_model = "Llama-2-mental-health"

  tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  model = AutoModelForCausalLM.from_pretrained(
      base_model,
      trust_remote_code=True,
      low_cpu_mem_usage=True
  )

  model.config.use_cache = False
  model.config.pretraining_tp = 1

  model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

  training_arguments = TrainingArguments(
    output_dir='./tmp',
    do_train=True,
    do_eval=True,
    num_train_epochs=3,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    save_strategy="no",
    log_level="info",
    save_total_limit=2,
    bf16=True,
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["Wqkv", "fc1", "fc2"]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=training_dataset,  # Replace with your actual training dataset
    peft_config=peft_config,
    dataset_text_field="Text",
    max_seq_length=690,
    tokenizer=tokenizer,
    args=training_arguments,
)

## Conclusion
This project provided us with a comprehensive understanding of fine-tuning large language models and the challenges associated with it. By optimizing the model to work efficiently in a constrained environment, we developed crucial skills in machine learning and model deployment.


