import importlib.util
from pathlib import Path
import unittest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from hexgen2.prepare_comparison import native_layer

class ConversionTest(unittest.TestCase):
    def test_each_native_layer_matches_original_artifact_remapper(self):
        config=LlamaConfig(vocab_size=128,hidden_size=128,intermediate_size=256,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4)
        torch.manual_seed(13);state=LlamaForCausalLM(config).state_dict()
        root=Path(__file__).resolve().parents[1]/'hexgen/llama'
        def module(name,path):
            spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
        remap=module('original_remap',root/'load_model_parameters_utils/remap_state_dict.py')
        mapping=module('original_config',root/'llama_config_utils.py')
        expected=remap.remap_state_dict_hf_llama(state.copy(),mapping.llama_config_to_gpt2_config(config))
        for layer in range(2):
            actual=native_layer(state.__getitem__,layer,128,4)
            for key,value in actual.items():
                self.assertTrue(torch.equal(value,expected[f'transformer.layers.{layer}.'+key]),key)
