import json
from pathlib import Path
import tempfile
import unittest
from gym_anything_robotics.provider_limits import recovery_model_evidence

class RecoveryIdentityTests(unittest.TestCase):
    def test_secondary_usage_requires_current_primary_evidence(self):
        event={'modelUsage':{'claude-opus-5-5':{},'claude-haiku-4-5-20251001':{}}}
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'projects/work/session.jsonl';p.parent.mkdir(parents=True)
            def entry(model,at='2026-10-17T16:00:00Z',**extra):
                return dict(type='assistant',sessionId='session',timestamp=at,message={'model':model},**extra)
            def check(entries):
                p.write_text(''.join(json.dumps(e)+'\n' for e in entries))
                return recovery_model_evidence(event,'claude-opus-5-5','session',1792250000,config_dir=temp)
            self.assertIsNone(check([]))
            self.assertIsNone(check([entry('claude-opus-5-5','2026-10-16T16:00:00Z')]))
            self.assertIsNone(check([entry('claude-opus-5-5',isSidechain=True)]))
            self.assertIsNone(check([entry('claude-haiku-4-5-20251001')]))
            self.assertIsNone(check([entry('claude-opus-5-5'),entry('claude-haiku-4-5-20251001')]))
            got=check([entry('claude-opus-5-5'),entry('<synthetic>')])
            self.assertEqual(got['primary_models'],['claude-opus-5-5'])
            self.assertIn('claude-haiku-4-5-20251001',got['aggregate_models'])

    def test_exact_usage_unchanged(self):
        self.assertEqual(recovery_model_evidence({'modelUsage':{'expected':{}}},'expected','id',0),
                         {'aggregate_models':['expected']})

if __name__=='__main__':unittest.main()
