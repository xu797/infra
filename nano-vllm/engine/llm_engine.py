from time import perf_counter
from dataclasses import fields
from transformers import AutoTokenizer

from sampling_params import SamplingParams
from config import Config
from sequence import Sequence

class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

    def add_request(self, prompt: str | list[int], sample_param: SamplingParams):
        """往schedule中添加request"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sample_param)
        self.scheduler.add_sequence(seq)
        

    def exit(self):
        """"结束inference"""
        pass

    def is_finished(self):
        """判断inference是否结束"""
        pass

    def step(self):
        """
        inference, return num_tokens, token_ids
        """
        pass

    def generate(self, prompts: list[str] | list[list[int]], sample_params: SamplingParams | list[SamplingParams]):

        if not isinstance(sample_params, list):
            sample_params = [sample_params] * len(prompts)

        #add request
        for prompt, param in zip(prompts, sample_params):
            self.add_request(prompt, param)

        #step generate
        outputs = []
        prefill_throughput = 0
        decode_throughput = 0

        while not self.is_finished():
            t = perf_counter()
            num_tokens, output = self.step() #output:list[(seq_id, decode_ids)...]

            if num_tokens > 0:   #prefill
                prefill_throughput += num_tokens / (perf_counter() - t)
            else:
                decode_throughput += num_tokens / (perf_counter() - t)

            for seq_id, tokens in output: #只有当前seq结束decode，output返回的才不是空
                output[seq_id] = tokens

        outputs = [outputs[seq_id] for seq_id in sorted[outputs.keys()]]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
