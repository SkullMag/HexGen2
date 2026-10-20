import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import torch
from hexgen2.comparison_model import NativeStage


class OutputTest(unittest.TestCase):
    def test_batched_output_counts_and_reused_storage(self):
        stage=object.__new__(NativeStage);torch.nn.Module.__init__(stage)
        stage.output_buffer=torch.empty(16,dtype=torch.int64)
        stage.step=lambda ids,*args:(ids[:,0]+1,None)
        stream=Mock()
        with patch('torch.cuda.current_stream',return_value=stream):
            for first,count in [(torch.tensor([10,20]),4),(torch.tensor([7]),1),(torch.tensor([3,8]),2)]:
                state=SimpleNamespace(sequence_len_offset=50);observed=[]
                result=stage.decode(first,state,count,None,0,1,lambda token:observed.append(token.clone()))
                self.assertEqual(result,[[int(x)+i for i in range(count)] for x in first])
                self.assertEqual(len(observed),count)
                self.assertEqual(state.sequence_len_offset,50+count-1)
        self.assertEqual(stream.synchronize.call_count,3)
