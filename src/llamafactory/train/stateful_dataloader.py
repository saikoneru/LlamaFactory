import os
import torch
from datasets import IterableDataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import hashlib
import json



def build_stateful_data_config(data_args):
    return {
        "dataset": data_args.dataset,
        "streaming": data_args.streaming,
        "mix_strategy": data_args.mix_strategy,
        "interleave_probs": data_args.interleave_probs,
        "buffer_size": data_args.buffer_size,
        "cutoff_len": data_args.cutoff_len,
        "packing": data_args.packing,
        "neat_packing": data_args.neat_packing,
    }


def get_stateful_data_fingerprint(config):
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class LLaMAFactoryStatefulDataLoader(StatefulDataLoader):
    def set_epoch(self, epoch):
        if getattr(self, "_skip_next_set_epoch", False):
            self._skip_next_set_epoch = False
            return

        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
            
            
def validate_stateful_dataloader_support(trainer):
    if getattr(trainer.args, "deepspeed", None):
        raise ValueError("StatefulDataLoader is not currently supported with DeepSpeed.")

    fsdp = getattr(trainer.args, "fsdp", None)
    if fsdp:
        raise ValueError("StatefulDataLoader is not currently supported with FSDP.")


def create_stateful_train_dataloader(trainer):
    validate_stateful_dataloader_support(trainer)
    train_dataset = trainer.train_dataset
    sampler = None

    generator = torch.Generator()
    generator.manual_seed(trainer.args.seed + trainer.args.process_index)

    if isinstance(train_dataset, IterableDataset):
        if trainer.args.world_size > 1:
            train_dataset = split_dataset_by_node(
                train_dataset,
                rank=trainer.args.process_index,
                world_size=trainer.args.world_size,
            )
    else:
        if trainer.args.world_size > 1:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=trainer.args.world_size,
                rank=trainer.args.process_index,
                shuffle=not trainer.finetuning_args.disable_shuffling,
                seed=trainer.args.seed,
                drop_last=trainer.args.dataloader_drop_last,
            )
        elif trainer.finetuning_args.disable_shuffling:
            sampler = SequentialSampler(train_dataset)
        else:
            sampler = RandomSampler(train_dataset, generator=generator)

    dataloader_params = {
        "batch_size": trainer._train_batch_size,
        "collate_fn": trainer.data_collator,
        "num_workers": trainer.args.dataloader_num_workers,
        "pin_memory": trainer.args.dataloader_pin_memory,
        "persistent_workers": trainer.args.dataloader_persistent_workers,
        "drop_last": trainer.args.dataloader_drop_last,
        "generator": generator,
    }

    if sampler is not None:
        dataloader_params["sampler"] = sampler

    if trainer.args.dataloader_num_workers > 0:
        dataloader_params["prefetch_factor"] = trainer.args.dataloader_prefetch_factor

    multiprocessing_context = getattr(trainer.args, "dataloader_multiprocessing_context", None)
    if multiprocessing_context is not None:
        dataloader_params["multiprocessing_context"] = multiprocessing_context

    return LLaMAFactoryStatefulDataLoader(train_dataset, **dataloader_params)
    
def save_stateful_dataloader_state(trainer, output_dir):
    dataloader = getattr(trainer, "_stateful_train_dataloader", None)
    if dataloader is None:
        raise RuntimeError("StatefulDataLoader has not been initialized.")
        
    data_config = getattr(trainer, "_stateful_data_config", None)
    if data_config is None:
        raise RuntimeError("StatefulDataLoader data configuration has not been initialized.")
    
    data_fingerprint = get_stateful_data_fingerprint(data_config)

    state = {
        "version": 1,
        "dataloader": dataloader.state_dict(),
        "rank": trainer.args.process_index,
        "world_size": trainer.args.world_size,
        "num_workers": trainer.args.dataloader_num_workers,
        "batch_size": trainer._train_batch_size,
        "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
        "seed": trainer.args.seed,
        "drop_last": trainer.args.dataloader_drop_last,
        "data_config": data_config,
        "data_fingerprint": data_fingerprint,
    }

    os.makedirs(output_dir, exist_ok=True)
    state_file = os.path.join(output_dir, f"dataloader_state_rank_{trainer.args.process_index}.pt")
    torch.save(state, state_file)


def load_stateful_dataloader_state(trainer, dataloader, checkpoint_dir):
    state_file = os.path.join(checkpoint_dir, f"dataloader_state_rank_{trainer.args.process_index}.pt")
    if not os.path.isfile(state_file):
        raise FileNotFoundError(
            f"StatefulDataLoader state for rank {trainer.args.process_index} was not found in {checkpoint_dir}."
        )

    state = torch.load(state_file, map_location="cpu", weights_only=False)

    version = state.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported StatefulDataLoader checkpoint version: {version}.")

    expected = {
        "rank": trainer.args.process_index,
        "world_size": trainer.args.world_size,
        "num_workers": trainer.args.dataloader_num_workers,
        "batch_size": trainer._train_batch_size,
        "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
        "seed": trainer.args.seed,
        "drop_last": trainer.args.dataloader_drop_last,
    }

    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"StatefulDataLoader checkpoint is incompatible: "
                f"{key}={state.get(key)} in checkpoint, current value={value}."
            )
            
    data_config = getattr(trainer, "_stateful_data_config", None)
    if data_config is None:
        raise RuntimeError("StatefulDataLoader data configuration has not been initialized.")
    
    data_fingerprint = get_stateful_data_fingerprint(data_config)
    saved_fingerprint = state.get("data_fingerprint")
    
    if saved_fingerprint is not None and saved_fingerprint != data_fingerprint:
        raise ValueError(
            "StatefulDataLoader checkpoint is incompatible with the current data configuration.\n"
            f"Checkpoint: {state.get('data_config')}\n"
            f"Current: {data_config}"
        )

    dataloader.load_state_dict(state["dataloader"])
    dataloader._skip_next_set_epoch = True
    return True