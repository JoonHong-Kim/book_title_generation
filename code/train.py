from transformers import AutoTokenizer, BartForConditionalGeneration, AutoConfig, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from data import KoBARTDataset
import torch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = AutoConfig.from_pretrained("gogamza/kobart-base-v2")
    dataset = pd.read_csv("/opt/ml/code/book_title_generation/data/all_data.csv")
    tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2",use_fast=True)
    model = BartForConditionalGeneration.from_pretrained(
        "gogamza/kobart-base-v2", config=config
    )
    
    model.to(device)
    train_dataset, dev_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )

    train_dataset = KoBARTDataset(train_dataset, tokenizer)
    dev_dataset = KoBARTDataset(dev_dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=3,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=10,  # total number of training epochs
        learning_rate=5e-5,  # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        #run_name="book_title_generation_v1",
        #report_to="wandb",
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        load_best_model_at_end=True,
    )
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )
    trainer.train()
    model.save_pretrained("./best_model")
    print("end")


if __name__ == "__main__":
    main()
