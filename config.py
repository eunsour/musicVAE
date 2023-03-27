drum_mapping = {
    36: 36,
    38: 38,
    40: 38,
    37: 38,
    48: 50,
    50: 50,
    45: 47,
    47: 47,
    43: 43,
    58: 43,
    46: 46,
    26: 46,
    42: 42,
    22: 42,
    44: 42,
    49: 49,
    55: 49,
    57: 49,
    52: 49,
    51: 51,
    59: 51,
    53: 51,
}

paper_to_idx = {key: idx for idx, key in enumerate(set(drum_mapping.values()))}
roland_to_idx = {key: paper_to_idx[value] for key, value in drum_mapping.items()}
num_class = len(set(roland_to_idx.values()))

params = {
    "paper_to_idx": paper_to_idx,
    "roland_to_idx": roland_to_idx,
    "num_class": num_class,
    "num_bars": 4,
    "num_units": 4,
    "num_sequence": 16,
    "path_model_trained": "./models/model_trained.pth",
    "train_batch_size": 256,
    "train_ratio": 0.75,
    "is_training_from_checkpoint": True,
}
