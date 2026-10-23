"""Offline checks for UniLIP component loss accounting and shuffled loading."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from torch.utils.data import RandomSampler
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

from llamafactory.hparams import FinetuningArguments, TrainingArguments
from llamafactory.train.callbacks import LogCallback
from llamafactory.train.sft.unilip_trainer import UniLIPSeq2SeqTrainer


@dataclass
class LossOutput(ModelOutput):
    loss: torch.Tensor = None
    text_loss: torch.Tensor = None
    image_loss: torch.Tensor = None


class TinyComposite(torch.nn.Module):
    accepts_loss_kwargs = False

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.config = PretrainedConfig()
        self.vision_decoder = SimpleNamespace(spec={"image_loss_weight": 0.3})

    def forward(self, input_ids, labels=None, target_images=None, image_batch_indices=None):
        text_loss = self.scale * input_ids.float().mean() if labels is not None else None
        image_loss = self.scale * target_images.mean() if target_images is not None else None
        loss = 0 if text_loss is None else 1.7 * text_loss
        if image_loss is not None:
            loss = loss + 0.3 * image_loss
        return LossOutput(loss=loss, text_loss=text_loss, image_loss=image_loss)


@pytest.fixture
def trainer(tmp_path):
    return UniLIPSeq2SeqTrainer(
        model=TinyComposite(),
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to="none", disable_tqdm=True,
            use_stateful_dataloader=True, dataloader_pin_memory=False,
        ),
        finetuning_args=FinetuningArguments(),
        processor=None,
        tokenizer=None,
    )


def inputs(*groups):
    return {
        "input_ids": torch.ones(4, 1, dtype=torch.long),
        "labels": torch.ones(4, 1, dtype=torch.long),
        "unilip_image_groups": [
            {"target_images": torch.full((count, 3, height, height), value),
             "image_batch_indices": torch.arange(count)}
            for count, height, value in groups
        ],
    }


@pytest.mark.parametrize("groups,expected_image", [
    ([], 0.0), ([(4, 2, 3.0)], 6.0), ([(1, 2, 3.0), (3, 4, 5.0)], 9.0),
])
def test_component_losses_and_gradients(trainer, groups, expected_image):
    # This also checks positional return_outputs and counts the LM loss only once.
    loss, _ = trainer.compute_loss(trainer.model, inputs(*groups), True)
    expected = 1.7 * 2.0 + 0.3 * expected_image
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert trainer.model.scale.grad.item() == pytest.approx(expected / 2)
    trainer.log({"loss": loss.item()})
    logged = trainer.state.log_history[-1]
    assert logged["text_loss"] == pytest.approx(2.0)
    assert logged["image_loss"] == pytest.approx(expected_image)
    assert not trainer._unilip_loss_totals.requires_grad


def test_interval_averaging_reset_and_eval_isolation(trainer):
    trainer.compute_loss(trainer.model, inputs((4, 2, 3.0)))
    trainer.compute_loss(trainer.model, inputs())
    trainer.model.eval()
    with torch.no_grad():
        trainer.compute_loss(trainer.model, inputs((4, 2, 100.0)))
    trainer.log({"eval_loss": 100.0})
    assert "image_loss" not in trainer.state.log_history[-1]
    trainer.log({"loss": 0.0})
    assert trainer.state.log_history[-1]["image_loss"] == pytest.approx(3.0)
    trainer.model.train()
    trainer.compute_loss(trainer.model, inputs((4, 2, 5.0)))
    trainer.log({"loss": 0.0})
    assert trainer.state.log_history[-1]["image_loss"] == pytest.approx(10.0)


def test_cpu_training_logs_components_and_shuffles(trainer, tmp_path):
    trainer.train_dataset = Dataset.from_dict({"input_ids": [[1]] * 16, "labels": [[1]] * 16})
    trainer.data_collator = lambda features: inputs((len(features), 2, 3.0))
    trainer.args.remove_unused_columns = False
    trainer.args.max_steps = 2
    trainer.args.per_device_train_batch_size = 2
    trainer.args.gradient_accumulation_steps = 2
    trainer.args.logging_steps = 1
    trainer.args.save_strategy = "no"
    trainer._stateful_data_config = {}
    trainer.add_callback(LogCallback())
    loader = trainer.get_train_dataloader()
    assert isinstance(loader.sampler, RandomSampler)
    order = list(loader.sampler)
    assert sorted(order) == list(range(16))
    assert order != list(range(16))
    trainer.train()
    logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert len(logs) == 2
    for entry in logs:
        assert entry["text_loss"] > 0
        assert entry["image_loss"] == pytest.approx(3 * entry["text_loss"])
        assert entry["loss"] == pytest.approx(1.7 * entry["text_loss"] + 0.3 * entry["image_loss"], abs=1e-3)
    saved = [json.loads(line) for line in (tmp_path / "trainer_log.jsonl").read_text().splitlines()]
    assert len([entry for entry in saved if "text_loss" in entry and "image_loss" in entry]) == 2


def test_projector_only_optimizes_image_loss_without_constant_language_loss(trainer):
    trainer.finetuning_args.unilip_projector_only = True
    loss = trainer.compute_loss(trainer.model, inputs((4, 2, 3.0)))
    assert loss.item() == pytest.approx(.3 * 6.0)
    loss.backward()
    assert trainer.model.scale.grad.item() == pytest.approx(.3 * 3.0)
    trainer.log({'loss': loss.item()})
    assert trainer.state.log_history[-1]['text_loss'] == 0
    assert trainer.state.log_history[-1]['image_loss'] == pytest.approx(6.)
    with pytest.raises(ValueError, match='requires image targets'):
        trainer.compute_loss(trainer.model, inputs())


@pytest.mark.parametrize('groups,expected_image', [([(4, 2, 3.)], 6.), ([(1, 2, 3.), (3, 4, 5.)], 9.)])
def test_bridge_only_uses_image_loss_for_single_and_multiple_buckets(trainer, groups, expected_image):
    trainer.finetuning_args.unilip_bridge_only = True
    loss, output = trainer.compute_loss(trainer.model, inputs(*groups), return_outputs=True)
    assert loss.item() == pytest.approx(.3 * expected_image)
    loss.backward()
    assert trainer.model.scale.grad.item() == pytest.approx(.3 * expected_image / 2)
    trainer.log({'loss': loss.item()})
    assert trainer.state.log_history[-1]['text_loss'] == 0
    assert trainer.state.log_history[-1]['image_loss'] == pytest.approx(expected_image)


def test_high_noise_sampling_is_fresh_shared_and_scaled_to_scheduler(trainer):
    trainer.finetuning_args.unilip_timestep_sampling = 'high_noise'
    decoder = trainer.model.vision_decoder
    decoder.spec['scheduler_config'] = {'num_train_timesteps': 2000}
    decoder.encode_images = lambda images: images * 3
    targets = torch.ones(4, 3, 2, 2, requires_grad=True)
    previous = None
    for step, expected in enumerate([2000, 2000, 2000, 1800, 1500, 1000, 500, 200]):
        trainer.state.global_step = step
        values = trainer._image_forward_kwargs(trainer.model, {'target_images': targets})
        assert 'target_images' not in values
        assert values['timesteps'].tolist() == [expected] * 4
        assert not values['latents'].requires_grad
        torch.testing.assert_close(values['latents'], targets.detach() * 3)
        assert torch.equal(values['noise'], values['noise'][:1].expand_as(targets))
        if previous is not None:
            assert not torch.equal(previous, values['noise'])
        previous = values['noise'].clone()
