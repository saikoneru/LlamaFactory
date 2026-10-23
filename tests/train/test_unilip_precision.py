"""Regression checks for optimizer precision after UniLIP trainability setup."""

from types import SimpleNamespace

import pytest
import torch

from llamafactory.train.sft import unilip_training


def make_model():
    model = torch.nn.Module()
    model.connector = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
    model.connector.weight.data.fill_(1)
    model.frozen_tokenizer = torch.nn.Linear(2, 2).to(torch.bfloat16).requires_grad_(False)
    return model


def test_upcast_preserves_small_updates_and_frozen_weights(monkeypatch):
    monkeypatch.setattr(unilip_training, 'is_deepspeed_zero3_enabled', lambda: False)
    model = make_model()
    frozen = model.frozen_tokenizer.weight.detach().clone()
    # At this LR, direct BF16 storage cannot represent the Adam update.
    baseline = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([baseline], lr=5e-5, weight_decay=0)
    baseline.sum().backward()
    optimizer.step()
    assert torch.equal(baseline, torch.ones_like(baseline))

    unilip_training._upcast_unilip_trainable_parameters(model, SimpleNamespace())
    assert model.connector.weight.dtype == torch.float32
    assert model.frozen_tokenizer.weight.dtype == torch.bfloat16
    assert torch.equal(model.frozen_tokenizer.weight, frozen)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=0)
    model.connector.weight.sum().backward()
    optimizer.step()
    torch.testing.assert_close(model.connector.weight, torch.full((2, 2), 1-5e-5), atol=1e-7, rtol=0)
    assert optimizer.state[model.connector.weight]['exp_avg'].dtype == torch.float32


@pytest.mark.parametrize('pure_bf16,use_badam,zero3', [(True,False,False),(False,True,False),(False,False,True)])
def test_preserves_explicit_or_managed_precision(monkeypatch, pure_bf16, use_badam, zero3):
    monkeypatch.setattr(unilip_training, 'is_deepspeed_zero3_enabled', lambda: zero3)
    model = make_model()
    unilip_training._upcast_unilip_trainable_parameters(model, SimpleNamespace(pure_bf16=pure_bf16,use_badam=use_badam))
    assert model.connector.weight.dtype == torch.bfloat16


def test_projector_only_keeps_gradient_through_frozen_connector():
    model = torch.nn.Module()
    model.qwen = torch.nn.Linear(3, 3)
    model.vision_decoder = torch.nn.Module()
    decoder = model.vision_decoder
    decoder.spec = {'conditioning_projection_mode': 'shared'}
    decoder.output_head = torch.nn.Linear(3, 3)
    decoder.llm_connector = torch.nn.Linear(3, 3)
    decoder.projector = torch.nn.Linear(3, 3)
    decoder.dit = torch.nn.Linear(3, 3)
    decoder.latent_queries = torch.nn.Embedding(2, 3)
    unilip_training._configure_projector_only(model, decoder)
    trainable = {n for n,p in model.named_parameters() if p.requires_grad}
    assert trainable == {'vision_decoder.output_head.weight', 'vision_decoder.output_head.bias'}
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    result = decoder.dit(decoder.projector(decoder.llm_connector(decoder.output_head(model.qwen(torch.ones(2,3))))))
    result.square().sum().backward()
    assert decoder.output_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for n,p in model.named_parameters() if n not in trainable)
    optimizer.step()
    assert not torch.equal(before['vision_decoder.output_head.weight'], decoder.output_head.weight)
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters() if n not in trainable)


@pytest.mark.parametrize('mode', ['disabled', 'lora', 'full'])
def test_projector_only_rejects_incompatible_mode(mode):
    from llamafactory.hparams import FinetuningArguments
    with pytest.raises(ValueError, match='requires unilip_train_mode: frozen'):
        FinetuningArguments(unilip_projector_only=True, unilip_train_mode=mode)


def test_freezing_pretrained_bridge_preserves_student_trainability():
    decoder = torch.nn.Module()
    decoder.output_head = torch.nn.Linear(3, 3)
    decoder.latent_queries = torch.nn.Embedding(2, 3)
    decoder.llm_connector = torch.nn.Linear(3, 3)
    decoder.projector = torch.nn.Linear(3, 3)
    omni_lora = torch.nn.Parameter(torch.ones(3))
    unilip_training._freeze_pretrained_bridge(decoder)
    assert omni_lora.requires_grad
    assert all(p.requires_grad for p in decoder.output_head.parameters())
    assert decoder.latent_queries.weight.requires_grad
    assert not any(p.requires_grad for p in decoder.llm_connector.parameters())
    assert not any(p.requires_grad for p in decoder.projector.parameters())


def test_bridge_only_freezes_everything_else_but_keeps_gradient_through_dit():
    model = torch.nn.Module()
    model.qwen = torch.nn.Linear(3, 3)
    model.vision_decoder = torch.nn.Module()
    decoder = model.vision_decoder
    decoder.spec = {'conditioning_projection_mode': 'shared'}
    for name in ['output_head', 'llm_connector', 'projector', 'dit', 'trigger_head']:
        setattr(decoder, name, torch.nn.Linear(3, 3))
    decoder.latent_queries = torch.nn.Embedding(2, 3)
    unilip_training._configure_bridge_only(model, decoder)
    allowed = ('vision_decoder.output_head.', 'vision_decoder.llm_connector.',
               'vision_decoder.projector.', 'vision_decoder.latent_queries.')
    assert all(p.requires_grad == n.startswith(allowed) for n,p in model.named_parameters())
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    hidden = model.qwen(torch.ones(2,3)) + decoder.latent_queries.weight
    result = decoder.dit(decoder.projector(decoder.llm_connector(decoder.output_head(hidden))))
    result.square().sum().backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(allowed))
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if n.startswith(allowed))
    optimizer.step()
    assert all(torch.equal(before[n], p) for n,p in model.named_parameters() if not n.startswith(allowed))


@pytest.mark.parametrize('mode', ['disabled', 'lora', 'full'])
def test_bridge_only_requires_frozen_dit(mode):
    from llamafactory.hparams import FinetuningArguments
    with pytest.raises(ValueError, match='unilip_bridge_only requires unilip_train_mode: frozen'):
        FinetuningArguments(unilip_bridge_only=True, unilip_train_mode=mode)


def test_prompt_only_omits_queries_and_preserves_trainable_bridge(tmp_path):
    import json

    model = torch.nn.Module()
    model.qwen = torch.nn.Linear(3, 3)
    model.vision_decoder = torch.nn.Module()
    decoder = model.vision_decoder
    decoder.spec = dict(conditioning_projection_mode='shared',
                        conditioning_sources=['image', 'prompt', 'queries'],
                        prompt_max_tokens=256, prompt_text_selection={'max_tokens': 256}, query_capacity=2)
    decoder._source_head_keys = {'image': 'source_0', 'prompt': 'source_1'}
    decoder.source_heads = torch.nn.ModuleDict()
    for name in ['output_head', 'llm_connector', 'projector', 'dit', 'trigger_head']:
        setattr(decoder, name, torch.nn.Linear(3, 3))
    decoder.latent_queries = torch.nn.Embedding(2, 3)
    unilip_training._configure_prompt_only(decoder)
    unilip_training._configure_bridge_only(model, decoder)
    allowed = ('vision_decoder.output_head.', 'vision_decoder.llm_connector.',
               'vision_decoder.projector.')
    assert all(p.requires_grad == n.startswith(allowed) for n,p in model.named_parameters())
    hidden = model.qwen(torch.ones(2, 3))
    result = decoder.dit(decoder.projector(decoder.llm_connector(decoder.output_head(hidden))))
    result.square().sum().backward()
    assert all(p.grad is not None for n,p in model.named_parameters() if n.startswith(allowed))
    assert decoder.latent_queries.weight.grad is None
    # Serialized routing must remain prompt-only for inference and supplemental resume checks.
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(decoder.spec))
    saved = json.loads(path.read_text())
    assert saved['conditioning_sources'] == ['prompt']
    unilip_training._check_checkpoint_spec(decoder, saved)


def test_prompt_only_rejects_an_incompatible_training_policy():
    from llamafactory.hparams import FinetuningArguments
    with pytest.raises(ValueError, match='unilip_prompt_only requires unilip_bridge_only'):
        FinetuningArguments(unilip_prompt_only=True, unilip_train_mode='frozen')
