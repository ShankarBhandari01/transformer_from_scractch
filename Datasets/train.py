import torch
import torch.nn as nn
import torch.multiprocessing as mp
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


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_idx('[SOS]')
    eos_idx = tokenizer_src.token_to_idx('[EOS]')

    # precompute the encoder output and reuse it  for every token we get from the decoder
    encoder_output = model.encoder(source, source_mask)
    # initialise the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
        # build the mask for the target (decoder input)
        decoder_mask = BilingualDataset.causal_mask(
            decoder_input.size(1)).type_as(source_mask).to(device)
        # calculate the output of the decoder
        out = model.decoder(encoder_output, source_mask,
                            decoder_input, decoder_mask)
        # get the next token
        prod = model.project(out[:, -1])
        # select the word with maximum prob
        _, next_word = torch.max(prod, dim=1)
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source_mask).fill_(
                next_word.item()).to(device)
        ], dim=1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_state,
    writer, nun_examples=2
):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    # size of the console window
    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for visualization"

            model_output = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]

            model_out_text = tokenizer_tgt.decode(
                model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # print in the console
            print_msg('-'*console_width)
            print_msg(f'SOURCE:{source_text}')
            print_msg(f'TARGET:{target_text}')
            print_msg(f'PREDICTED:{model_out_text}')

            if count == nun_examples:
                break


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

    # Use multiprocessing for data loading
    num_workers = 3  
    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=False)
    val_dataloader = DataLoader(
        val_ds, batch_size=1, num_workers=num_workers, pin_memory=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def build_model(config, vocab_src_len, vocab_tgt_len):
    model = Transformer.build_transformer(vocab_src_len, vocab_tgt_len,
                                          config['seq_len'], config['d_model'], config['seq_len'])
    return model


def train_model(config, rank=0):
    """Main training function. 'rank' can differentiate processes if needed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Process {rank}: Using device: {device}')
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config)

    model = build_model(config, tokenizer_src.get_vocab_size(),
                        tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(f"{config['experiment_name']}_proc{rank}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weight_file_path(config, config['preload'])
        print(f'Process {rank}: Preloading model from file: {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fun = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id(
        "[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):

        batch_iterator = tqdm(
            train_dataloader, desc=f"Proc {rank} Epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()

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
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                       config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weight_file_path(
            config, f'{epoch:02d}_proc{rank}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


def train_worker(config, rank):
    train_model(config, rank)


if __name__ == '__main__':
    print("Training started with multiprocessing...")
    warnings.filterwarnings('ignore')
    config = get_config()

    # Set start method for multiprocessing (important for macOS)
    mp.set_start_method('spawn', force=True)

    num_processes = 1  # Number of parallel processes to spawn; 

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train_worker, args=(config, rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Training completed across all processes.")
