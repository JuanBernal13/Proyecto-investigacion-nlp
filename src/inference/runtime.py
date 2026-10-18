"""Backend Transformers local, plantilla oficial, contexto estricto y medidas CUDA."""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import random
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.transformations.core import require


def resolve_device(requested: str) -> str:
    require(requested in {'auto', 'cpu', 'cuda'}, 'Dispositivo desconocido')
    if requested == 'cuda':
        require(torch.cuda.is_available(), 'CUDA solicitada pero no disponible')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested


class TransformersBackend:
    def __init__(self, model: dict[str, str], config: dict[str, Any], *, allow_download: bool = False) -> None:
        self.config = config
        self.device = resolve_device(config['device'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        kwargs = {'revision': model['revision'], 'local_files_only': not allow_download, 'trust_remote_code': False}
        self.tokenizer = AutoTokenizer.from_pretrained(model['id'], **kwargs)
        template = self.tokenizer.get_chat_template()
        require(bool(template), 'Falta la plantilla oficial de chat; no usar plantilla alternativa')
        start = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            model['id'], **kwargs, torch_dtype=dtype, attn_implementation=config['attention_implementation'],
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.generation = deepcopy(self.model.generation_config)
        self.generation.do_sample = False
        self.generation.max_new_tokens = config['max_new_tokens']
        self.generation.num_beams = 1
        self.generation.num_return_sequences = 1
        self.generation.use_cache = True
        self.generation.temperature = 1.0
        self.generation.top_p = 1.0
        self.generation.top_k = 50
        self.generation.repetition_penalty = 1.0
        if self.generation.pad_token_id is None:
            self.generation.pad_token_id = self.tokenizer.eos_token_id
        position_limit = getattr(self.model.config, 'max_position_embeddings', config['max_context_tokens'])
        require(position_limit >= config['max_context_tokens'], 'El modelo no soporta el contexto configurado')
        self.metadata = {
            'model': model, 'device': self.device, 'dtype': str(dtype),
            'gpu': torch.cuda.get_device_name(0) if self.device == 'cuda' else None,
            'gpu_total_memory': torch.cuda.get_device_properties(0).total_memory if self.device == 'cuda' else None,
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'chat_template': template, 'chat_template_sha256': hashlib.sha256(template.encode('utf-8')).hexdigest(),
            'tokenizer_sha256': hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode('utf-8')).hexdigest(),
            'generation_config': self.generation.to_dict(), 'attention_implementation': config['attention_implementation'],
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'tf32': torch.backends.cuda.matmul.allow_tf32,
        }
        self.load_seconds = time.perf_counter() - start

    def generate(self, task: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        # Un único mensaje user evita diferencias de instrucciones entre brazos/datasets.
        message = self.config['prompt'] + task['problem']
        encoded = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': message}], tokenize=True, add_generation_prompt=True,
            return_tensors='pt', return_dict=True, truncation=False,
        )
        ids = encoded['input_ids'][0].tolist()
        base = {'input_tokens': len(ids), 'input_token_ids': ids,
                'prompt_sha256': hashlib.sha256(message.encode('utf-8')).hexdigest()}
        if len(ids) + self.config['max_new_tokens'] > self.config['max_context_tokens']:
            return {**base, 'status': 'context_overflow', 'generation_attempted': False,
                    'reason': 'input_plus_reserved_output_exceeds_total_context'}
        encoded = encoded.to(self.device)
        tokenization_seconds = time.perf_counter() - start
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**encoded, generation_config=self.generation)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        allocated = torch.cuda.max_memory_allocated() if self.device == 'cuda' else None
        reserved = torch.cuda.max_memory_reserved() if self.device == 'cuda' else None
        generated = output[0, len(ids):].tolist()
        eos = self.generation.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else (eos or [])
        eos_reached = bool(generated and generated[-1] in eos_ids)
        return {**base, 'status': 'generated', 'generation_attempted': True,
                'output_token_ids': generated, 'output_tokens': len(generated),
                'text': self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                'finish_reason': 'eos' if eos_reached else 'length_or_other_stop',
                'reached_token_limit': len(generated) >= self.config['max_new_tokens'],
                'generation_seconds': elapsed, 'tokenization_and_transfer_seconds': tokenization_seconds,
                'tokens_per_second': len(generated)/elapsed if elapsed > 0 else None,
                'peak_cuda_allocated_bytes': allocated, 'peak_cuda_reserved_bytes': reserved,
                'model_load_seconds': self.load_seconds}

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
