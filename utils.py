import os
import numpy as np
import h5py
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import pickle
import math

def create_input_files(dataset, karpathy_json_path, captions_per_image, min_word_freq, output_folder, max_len=150):
    """
    Creates input files for training, validation, and test data.
    Modifications pour Transformer :
    - Augmentation de max_len à 150 (meilleure gestion des longues séquences)
    - Ajout de tokens spéciaux pour les Transformers
    """
    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    
    # Chargement des features images
    with open(os.path.join(output_folder, 'train36_imgid2idx.pkl'), 'rb') as j:
        train_data = pickle.load(j)
    with open(os.path.join(output_folder, 'val36_imgid2idx.pkl'), 'rb') as j:
        val_data = pickle.load(j)

    # Initialisation des structures de données
    train_image_captions = []
    val_image_captions = []
    test_image_captions = []
    train_image_det = []
    val_image_det = []
    test_image_det = []
    word_freq = Counter()

    # Nouveau : compteur pour statistiques
    total_captions = 0
    discarded_captions = 0

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            total_captions += 1
            if len(c['tokens']) > max_len:
                discarded_captions += 1
                continue
            word_freq.update(c['tokens'])
            captions.append(c['tokens'])

        if not captions:
            continue

        image_id = int(img['filename'].split('_')[2].lstrip("0").split('.')[0])

        # Gestion des splits
        if img['split'] in {'train', 'restval'}:
            target_data = train_data if img['filepath'] == 'train2014' else val_data
            if image_id in target_data:
                prefix = 't' if img['filepath'] == 'train2014' else 'v'
                train_image_det.append((prefix, target_data[image_id]))
                train_image_captions.append(captions)
        elif img['split'] == 'val':
            if image_id in val_data:
                val_image_det.append(("v", val_data[image_id]))
                val_image_captions.append(captions)
        elif img['split'] == 'test':
            if image_id in val_data:
                test_image_det.append(("v", val_data[image_id]))
                test_image_captions.append(captions)

    # Nouveau : affichage des statistiques
    print(f"Total captions: {total_captions}")
    print(f"Discarded captions (length > {max_len}): {discarded_captions}")
    print(f"Vocabulary size: {len(word_freq)}")

    # Création du vocabulaire avec tokens supplémentaires pour les Transformers
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    
    # Tokens spéciaux
    special_tokens = {
        '<pad>': 0,
        '<unk>': len(word_map) + 1,
        '<start>': len(word_map) + 2,
        '<end>': len(word_map) + 3,
        '<mask>': len(word_map) + 4  # Nouveau : pour l'apprentissage auto-supervisé
    }
    word_map.update(special_tokens)

    # Sauvegarde du vocabulaire
    base_filename = f"{dataset}_{captions_per_image}_cap_per_img_{min_word_freq}_min_word_freq"
    os.makedirs(output_folder, exist_ok=True)
    
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j, indent=4)

    # Fonction pour créer les données encodées
    def process_split(impaths, imcaps, split_name):
        enc_captions = []
        caplens = []
        
        for i, path in enumerate(tqdm(impaths, desc=f"Processing {split_name}")):
            # Échantillonnage des légendes
            available_captions = imcaps[i]
            if len(available_captions) < captions_per_image:
                captions = available_captions + [
                    choice(available_captions) 
                    for _ in range(captions_per_image - len(available_captions))
                ]
            else:
                captions = sample(available_captions, k=captions_per_image)
            
            # Encodage
            for c in captions:
                enc_c = [word_map['<start>']] + \
                       [word_map.get(word, word_map['<unk>']) for word in c] + \
                       [word_map['<end>']] + \
                       [word_map['<pad>']] * (max_len - len(c))
                
                enc_captions.append(enc_c)
                caplens.append(len(c) + 2)  # +2 pour <start> et <end>

        # Sauvegarde
        with open(os.path.join(output_folder, split_name + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_captions, j)
        
        with open(os.path.join(output_folder, split_name + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)
        
        # Sauvegarde des détections images
        with open(os.path.join(output_folder, split_name + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
            json.dump(impaths, j)

    # Traitement des différents splits
    process_split(train_image_det, train_image_captions, 'TRAIN')
    process_split(val_image_det, val_image_captions, 'VAL')
    process_split(test_image_det, test_image_captions, 'TEST')

def init_embedding(embeddings):
    """Initialisation des embeddings avec Xavier uniform pour les Transformers"""
    nn.init.xavier_uniform_(embeddings.weight)
    if embeddings.padding_idx is not None:
        with torch.no_grad():
            embeddings.weight[embeddings.padding_idx].fill_(0)

def save_checkpoint(data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer,
                   scheduler, bleu4, is_best, checkpoint_dir='checkpoints'):
    """Sauvegarde améliorée avec gestion du scheduler"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'decoder': decoder.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'vocab_size': decoder.vocab_size,
        'decoder_dim': decoder.decoder_dim,
        'attention_dim': decoder.attention_dim,
        'embed_dim': decoder.embed_dim,
        'dropout': decoder.dropout
    }
    
    filename = os.path.join(checkpoint_dir, f'checkpoint_{data_name}_epoch{epoch}.pth.tar')
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(checkpoint_dir, f'BEST_checkpoint_{data_name}.pth.tar')
        torch.save(state, best_filename)
        print(f"New best model saved to {best_filename}")

def load_checkpoint(checkpoint_path, decoder, decoder_optimizer=None, scheduler=None):
    """Chargement du checkpoint avec gestion des erreurs"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    decoder.load_state_dict(checkpoint['decoder'])
    
    if decoder_optimizer is not None and 'decoder_optimizer' in checkpoint:
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return {
        'epoch': checkpoint['epoch'],
        'bleu4': checkpoint['bleu-4'],
        'epochs_since_improvement': checkpoint['epochs_since_improvement']
    }

class AverageMeter:
    """Improved with smoothing and history tracking"""
    def __init__(self, window_size=100):
        self.reset()
        self.window_size = window_size
        self.history = []
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
    def smoothed_avg(self):
        return np.mean(self.history) if self.history else 0

def adjust_learning_rate(optimizer, shrink_factor, min_lr=1e-6):
    """LR adjustment with minimum threshold"""
    print(f"\nReducing learning rate by factor {shrink_factor}")
    for param_group in optimizer.param_groups:
        new_lr = max(param_group['lr'] * shrink_factor, min_lr)
        param_group['lr'] = new_lr
    print(f"New learning rate: {optimizer.param_groups[0]['lr']:.2e}")

def accuracy(scores, targets, k, ignore_index=-100):
    """Top-k accuracy with ignore index support"""
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    
    # Create mask for valid targets
    mask = targets != ignore_index
    targets_masked = targets[mask]
    ind_masked = ind[mask.repeat(1, k).view(-1, k)]
    
    if len(targets_masked) == 0:
        return 0.0
    
    correct = ind_masked.eq(targets_masked.view(-1, 1).expand_as(ind_masked))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / len(targets_masked))

def create_masks(tgt, pad_idx):
    """Create masks for Transformer decoder"""
    # Mask pour le padding
    tgt_pad_mask = (tgt == pad_idx)
    
    # Mask pour l'auto-régression
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), diagonal=0).bool()
    tgt_sub_mask = tgt_sub_mask.to(tgt.device)
    
    # Combine masks
    tgt_mask = tgt_pad_mask.unsqueeze(1) | (~tgt_sub_mask)
    
    return tgt_pad_mask, tgt_mask

def clip_gradient(optimizer, grad_clip):
    """Clip gradients at specified value""" 
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def set_seed(seed_value=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
