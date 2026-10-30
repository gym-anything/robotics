import unittest
from unittest.mock import patch
from gym_anything_robotics.errors import ActionRejectedError
from tests.test_http_action_rejection import ActionRejectionHTTPTest,COMMAND

class PartialBatchTests(unittest.TestCase):
    setUp=ActionRejectionHTTPTest.setUp
    tearDown=ActionRejectionHTTPTest.tearDown
    request=ActionRejectionHTTPTest.request
    json=staticmethod(ActionRejectionHTTPTest.json)

    def test_known_partial_is_reported_cached_and_next_motion_works(self):
        self.server.service.recover_known_partial_batches=True
        original=self.backend.act
        calls=[]
        def act(action):
            calls.append(action)
            if len(calls)==2:raise ActionRejectedError('IK unreachable; private diagnostics')
            return original(action)
        principal=self.server.keys.authenticate('Bearer policy-token')
        payload={'actions':[COMMAND,COMMAND,COMMAND]}
        with patch.object(self.backend,'act',side_effect=act):
            first=self.server.service.action(payload,idempotency_key='one',principal=principal)
            second=self.server.service.action(payload,idempotency_key='one',principal=principal)
        self.assertEqual(first,second)
        self.assertEqual(self.backend.tick,1)
        self.assertEqual(len(calls),2)
        self.assertTrue(first['committed'])
        self.assertTrue(first['partial'])
        self.assertEqual(first['unexecuted_action_indices'],[1,2])
        self.assertNotIn('private diagnostics',str(first))
        self.assertIsNotNone(first['observation'])
        self.assertTrue(self.server.service.health()['ready'])
        self.server.service.action(COMMAND,idempotency_key='two',principal=principal)
        self.assertEqual(self.backend.tick,2)

    def test_unknown_failure_still_faults(self):
        self.server.service.recover_known_partial_batches=True
        original=self.backend.act
        def act(action):
            if self.backend.tick:raise ValueError('unknown physics failure')
            return original(action)
        with patch.object(self.backend,'act',side_effect=act),self.assertRaises(ValueError):
            self.server.service.action({'actions':[COMMAND,COMMAND]},idempotency_key='one')
        self.assertFalse(self.server.service.health()['ready'])

    def test_unexpected_error_while_a_command_runs_is_a_server_error_not_an_invalid_request(self):
        # syringe_handling's trace raised KeyError('tick') after physics had advanced; the generic handler answered 400
        # "Invalid request; consult /v1/help." to the policy for a command that was valid.
        import urllib.error
        original=self.backend.act
        def act(action):
            if self.backend.tick:
                self.backend.tick+=1
                raise KeyError('secret-internal-detail')
            return original(action)
        for token in ('policy-token','test-token'):
            with self.subTest(token=token):
                self.backend.tick=0
                with patch.object(self.backend,'act',side_effect=act),self.assertRaises(urllib.error.HTTPError) as caught:
                    self.request('/v1/action',method='POST',body={'actions':[COMMAND,COMMAND,COMMAND]},token=token)
                self.assertEqual(caught.exception.code,500)
                error=self.json(caught.exception)
                self.assertEqual(error['error_type'],'ActionExecutionError')
                self.assertIn('KeyError',error['error'])
                self.assertIn('command 2 of 3',error['error'])
                self.assertNotIn('Invalid request',error['error'])
                self.assertEqual((error['completed_actions'],error['failed_action_index']),(1,1))
                self.assertEqual(caught.exception.headers['X-GAR-Request-ID'],error['request_id'])
                self.assertEqual('secret-internal-detail' in error['error'],token=='test-token')
                self.assertEqual(self.backend.tick,2)                    # the third command did not run
                records=self.server.service.history.store.list()['records']
                record=self.server.service.history.store.get(records[-1]['id'])
                self.assertEqual(record['error']['type'],'KeyError')
                self.assertTrue(self.server.service.health()['ready'])     # fault latching unchanged for this kind

    def test_errors_before_any_command_runs_keep_their_answer(self):
        import urllib.error
        for body,status in (({'actions':[]},400),({'type':'cartesian_delta','arm':'left'},400)):
            with self.subTest(body=body),self.assertRaises(urllib.error.HTTPError) as caught:
                self.request('/v1/action',method='POST',body=body,token='policy-token')
            self.assertEqual(caught.exception.code,status)
            self.assertNotEqual(self.json(caught.exception)['error_type'],'ActionExecutionError')

if __name__=='__main__':unittest.main()
