import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from config.config import get_weight_file_path, get_config
from Transfomer.transformer import Transformer
from Datasets.BilingualDataset import BilingualDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import re


def extract_from_instruction(example):
    pattern = r'\[INST\] Please translate "(.*?)" into Nepali \[/INST\] Sure! here is the translated text into Nepali: (.*?)</s>'
    match = re.search(pattern, example['text'])
    if match:
        return {'source': match.group(1).strip(), 'target': match.group(2).strip()}
    return {'source': '', 'target': ''}


def get_all_sentences(ds, field):
    for item in ds:
        if item[field]:
            yield item[field]


def get_or_build_tokenizer(config, ds, field):
    tokenizer_path = Path(config['tokenizer_file'].format(field))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, field), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    # Load and extract sentences from Hugging Face dataset
    ds_raw = load_dataset(
        "ashokpoudel/English-Nepali-Translation-Instruction-Dataset", split="train")
    ds_raw = ds_raw.map(extract_from_instruction)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, "source")
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, "target")

    # Train-validation split
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_tgt, "source", "target", config['seq_len'])
    val_ds = BilingualDataset(
        val_ds_raw, tokenizer_src, tokenizer_tgt, "source", "target", config['seq_len'])

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def build_model(config, vocab_src_len, vocab_tgt_len):
    model = Transformer.build_transformer(vocab_src_len, vocab_tgt_len,
                                          config['seq_len'], config['d_model'], config['seq_len'])
    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config)

    model = build_model(config, tokenizer_src.get_vocab_size(),
                        tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weight_file_path(config, config['preload'])
        print(f'Preloading model from file: {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fun = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id(
        "[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask)
            projection_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fun(projection_output.view(-1,
                            tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():.3f}'})

            writer.add_scalar("Train Loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        model_filename = get_weight_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    print("Training started...")
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
