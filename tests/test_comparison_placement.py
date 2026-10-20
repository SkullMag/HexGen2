import json,tempfile,unittest
from pathlib import Path
from hexgen2.comparison_launch import placement

class PlacementTest(unittest.TestCase):
    def test_two_tail_stages_can_share_a_vm_without_sharing_a_gpu(self):
        rows=[{'rank':r,'node':'gpu2' if r%2 else 'gpu'+str(r//2),'local_device':r//2 if r%2 else 0,'bind':'10.0.0.'+str(r%2+1)} for r in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'placement.json';path.write_text(json.dumps({'ranks':rows}))
            self.assertEqual([r for r,_,_ in placement(path,'gpu2')],[1,3])
            rows[3]['local_device']=0;path.write_text(json.dumps({'ranks':rows}))
            with self.assertRaises(ValueError):placement(path,'gpu2')
