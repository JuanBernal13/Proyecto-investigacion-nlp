"""Transformers real con un modelo diminuto aleatorio local: no resultado científico."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from src.inference.runtime import TransformersBackend, resolve_device
from src.inference.engine import execute

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def tiny_model(tmp_path_factory: Any) -> dict[str, str]:
    directory = tmp_path_factory.mktemp('random_test_model')
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, '[PAD]': 2, '2': 3, '+': 4, '3': 5}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', eos_token='[EOS]', pad_token='[PAD]')
    fast.chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}{% if add_generation_prompt %} assistant:{% endif %}"
    fast.save_pretrained(directory)
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_positions=256, n_embd=16, n_layer=1, n_head=2,
                                     eos_token_id=1, pad_token_id=2, bos_token_id=1))
    model.save_pretrained(directory, safe_serialization=True)
    return {'id': str(directory), 'revision': 'local-random-fixture-only'}


def small_config(device: str = 'cpu') -> dict[str, Any]:
    config = json.loads((ROOT / 'configs/inference.json').read_text(encoding='utf-8'))
    return {**config, 'device': device, 'max_context_tokens': 256, 'max_new_tokens': 4}


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_transformers_generation_and_measurements(tiny_model: dict[str, str], device: str) -> None:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA no disponible')
    backend = TransformersBackend(tiny_model, small_config(device))
    try:
        assert not backend.model.training
        assert all(not p.requires_grad for p in backend.model.parameters())
        result = backend.generate({'problem': '2 + 3'})
        assert result['status'] == 'generated'
        assert 0 < result['output_tokens'] <= 4
        assert result['generation_seconds'] > 0
        assert len(result['input_token_ids']) == result['input_tokens']
        assert result['input_tokens'] + result['output_tokens'] <= 256
        assert len(result['prompt_sha256']) == 64
        assert (result['peak_cuda_allocated_bytes'] is not None) == (device == 'cuda')
        if device == 'cuda':
            assert result['peak_cuda_allocated_bytes'] > 0
            assert result['peak_cuda_reserved_bytes'] >= result['peak_cuda_allocated_bytes']
    finally:
        backend.close()


def test_context_rejection_never_calls_generate(tiny_model: dict[str, str], monkeypatch: Any) -> None:
    backend = TransformersBackend(tiny_model, small_config())
    try:
        monkeypatch.setattr(backend.model, 'generate', lambda **kwargs: pytest.fail('No generar fuera del contexto'))
        result = backend.generate({'problem': '2 ' * 300})
        assert result['status'] == 'context_overflow' and result['generation_attempted'] is False
        assert result['input_tokens'] > 256  # No se truncó el original.
    finally:
        backend.close()


def test_generated_suffix_excludes_prompt_and_reports_eos(tiny_model: dict[str, str], monkeypatch: Any) -> None:
    backend = TransformersBackend(tiny_model, small_config())
    try:
        def fixed(**kwargs: Any) -> torch.Tensor:
            return torch.cat([kwargs['input_ids'], torch.tensor([[3, 1]])], dim=1)
        monkeypatch.setattr(backend.model, 'generate', fixed)
        result = backend.generate({'problem': '2 + 3'})
        assert result['text'] == '2' and result['output_token_ids'] == [3, 1]
        assert result['finish_reason'] == 'eos' and result['reached_token_limit'] is False
    finally:
        backend.close()


def test_cuda_not_assumed(monkeypatch: Any) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device('auto') == 'cpu'
    with pytest.raises(ValueError):
        resolve_device('cuda')


def test_full_engine_with_real_backend_can_resume(tiny_model: dict[str, str], tmp_path: Path) -> None:
    config = small_config()
    tasks = [{'variant_id': str(i), 'problem': '2 + 3', 'applicable': True} for i in range(2)]
    factory = lambda: TransformersBackend(tiny_model, config)
    output = tmp_path / 'tiny_run'
    assert execute(output, {'fixture_only': True}, tasks, factory)['status'] == 'incomplete'
    result = execute(output, {'fixture_only': True}, tasks, factory, resume=True)
    assert result['status'] == 'completed' and result['counts'] == {'generated': 2}


def test_offline_load_and_prompt_do_not_include_reference(tiny_model: dict[str, str], monkeypatch: Any) -> None:
    from src.inference import runtime
    original = runtime.AutoTokenizer.from_pretrained
    calls = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.AutoTokenizer, 'from_pretrained', capture)
    config = small_config()
    backend = TransformersBackend(tiny_model, config)
    messages = []
    original_template = backend.tokenizer.apply_chat_template

    def capture_template(chat: Any, **kwargs: Any) -> Any:
        messages.extend(chat)
        assert kwargs['truncation'] is False and kwargs['add_generation_prompt'] is True
        return original_template(chat, **kwargs)

    monkeypatch.setattr(backend.tokenizer, 'apply_chat_template', capture_template)
    try:
        backend.generate({'problem': '2 + 3', 'answer': 'DO_NOT_INCLUDE_REFERENCE'})
        assert messages == [{'role': 'user', 'content': config['prompt'] + '2 + 3'}]
        assert calls[0]['local_files_only'] is True and calls[0]['trust_remote_code'] is False
        assert calls[0]['revision'] == tiny_model['revision']
    finally:
        backend.close()


def test_length_limit_recorded_without_eos(tiny_model: dict[str, str], monkeypatch: Any) -> None:
    backend = TransformersBackend(tiny_model, small_config())
    try:
        monkeypatch.setattr(backend.model, 'generate', lambda **kwargs: torch.cat(
            [kwargs['input_ids'], torch.tensor([[3, 3, 3, 3]])], dim=1))
        result = backend.generate({'problem': '2 + 3'})
        assert result['reached_token_limit'] is True and result['finish_reason'] == 'length_or_other_stop'
    finally:
        backend.close()
