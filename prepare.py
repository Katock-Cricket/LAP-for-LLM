import os

from huggingface_hub import snapshot_download

weights_dir = 'weights'
dataset_dir = 'dataset'
dataset_conv_dir = 'dataset_conv'


# Download weights and dataset
def download_weights_and_dataset():
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.listdir(weights_dir):
        snapshot_download(repo_id="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
                          repo_type="model",
                          local_dir=weights_dir)

    if not os.listdir(dataset_dir):
        snapshot_download(repo_id="mlabonne/FineTome-100k",
                          repo_type="dataset",
                          local_dir=dataset_dir)


# Load model and tokenizer
def load_model_and_tokenizer():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=weights_dir,  # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    print("Model loaded!")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    print("PeFT model loaded!")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    print("Tokenizer loaded!")
    return model, tokenizer


# Convert dataset to HuggingFace's normal multiturn format ("role", "content") instead of ("from", "value")/ Llama-3 renders multi turn conversations
def convert_dataset_to_huggingface_format(tokenizer):
    if not os.path.exists(dataset_conv_dir):
        os.makedirs(dataset_conv_dir)

    if not os.listdir(dataset_conv_dir):
        from datasets import load_dataset
        from unsloth.chat_templates import standardize_sharegpt

        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in
                     convos]
            return {"text": texts, }

        dataset = load_dataset(dataset_dir, split="train")
        dataset = standardize_sharegpt(dataset)
        dataset = dataset.map(formatting_prompts_func, batched=True, )
        dataset.save_to_disk(dataset_conv_dir)


def load_converted_dataset():
    from datasets import load_from_disk
    dataset = load_from_disk(dataset_conv_dir)
    print("Dataset loaded!")
    return dataset


if __name__ == '__main__':
    download_weights_and_dataset()
    model, tokenizer = load_model_and_tokenizer()
    convert_dataset_to_huggingface_format(tokenizer)
    dataset = load_converted_dataset()
    print(dataset[5]["text"])
