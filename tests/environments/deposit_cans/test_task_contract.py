"""Non-simulating checks of the per-preset task contracts and the authored high_fill pile."""
import unittest
from dataclasses import replace
import numpy as np
from deposit_cans.config import Config
from deposit_cans.variants import PRESETS, preset, task_resolve
from deposit_cans import bag_pile, task

class ContractTests(unittest.TestCase):
    def test_contract_is_per_preset(self):
        self.assertEqual(task.task_contract(replace(Config(scene_revision=7),**PRESETS['high_fill'])),task.HIGH_FILL_CONTRACT)
        self.assertEqual(task.task_contract(replace(Config(scene_revision=7),**PRESETS['reference'])),'v1')
        # Configurations with other inventories keep the v1 checks.
        self.assertIsNone(preset(Config(target_count=3,tub_count=3,preload_count=48)))
        self.assertEqual(task.task_contract(Config(scene_revision=7,target_count=3,tub_count=3,preload_count=48)),'v1')
    def test_descriptions(self):
        self.assertEqual(task_resolve({'variant':'reference'})['description'],task.DESCRIPTION)
        # The eight-task catalogue's reference entry (explicit contract) states the job as high_fill does.
        self.assertEqual(task_resolve({'variant':'reference','contract':'deposit-tasks-v2'})['description'],
            task.HIGH_FILL_DESCRIPTION)
        text=task_resolve({'variant':'high_fill','seed':3})['description']
        self.assertEqual(text,task.HIGH_FILL_DESCRIPTION)
        for word in ('designated','floor','settled','lodged','Preserve'):self.assertNotIn(word,text)
        with self.assertRaises(ValueError):task.description('missing')
    def test_area_share_follows_insertion_in_every_orientation(self):
        from scipy.spatial.transform import Rotation
        from deposit_cans.geometry import can_mesh
        v,f,_=can_mesh(1,24,0.,quality_end=True)
        for turn in (Rotation.identity(),Rotation.from_euler('x',np.pi/2),Rotation.from_euler('x',np.pi/4)):
            tri=turn.apply(v)[f];centre=tri.mean(1)[:,2];lo,hi=tri[...,2].min(),tri[...,2].max()
            values=[task.shell_area_inside_share(tri,centre<lo+s*(hi-lo)) for s in (-.01,.25,.5,.75,1.01)]
            self.assertEqual(values[0],0.);self.assertEqual(values[-1],1.)
            self.assertTrue(all(x<y for x,y in zip(values,values[1:])))
            self.assertAlmostEqual(values[2],.5,delta=.08)   # half the can below the cut, any orientation

class RestWindowTests(unittest.TestCase):
    def run_motion(self,speed_m_s,seconds=25.,every=5,spike_tick=None):
        """One tracked centre moving at constant speed; returns the rest verdict at each sampled tick."""
        history=[];verdicts=[];quaternions=np.tile([0,0,0,1.],(2,1))
        for tick in range(0,int(seconds*240)+1,every):
            position=np.array([[speed_m_s*tick/240,0,0],[0,0,0],[0,0,0]])
            if tick==spike_tick:position=position+[.0,.0,.002]
            verdicts.append((tick,task.trailing_rest(history,tick,position,quaternions,240)[0]))
        return verdicts
    def test_slow_creep_stays_at_rest(self):
        # About 1 mm/s of creep must not interrupt rest, however long it lasts.
        for speed in (.0009,.001,.002,.009):
            self.assertTrue(all(ok for tick,ok in self.run_motion(speed) if tick>=240),speed)
    def test_motion_faster_than_the_window_limit_is_not_rest(self):
        self.assertFalse(any(ok for tick,ok in self.run_motion(.011)))
        self.assertFalse(any(ok for tick,ok in self.run_motion(.03)))
    def test_window_must_be_covered_and_small_pulses_do_not_veto(self):
        verdicts=dict(self.run_motion(0.,seconds=3.,every=1,spike_tick=400))
        self.assertFalse(verdicts[239]);self.assertTrue(verdicts[240]);self.assertTrue(verdicts[400]);self.assertTrue(verdicts[720])
    def test_rotation_limit(self):
        history=[];spin=lambda deg:np.vstack([[0,0,0,1.],__import__('scipy.spatial.transform',fromlist=['Rotation']).Rotation.from_euler('z',deg,degrees=True).as_quat()])
        position=np.zeros((3,3))
        task.trailing_rest(history,0,position,spin(0.),240)
        self.assertTrue(task.trailing_rest(history,240,position,spin(2.),240)[0])
        self.assertFalse(task.trailing_rest(history,480,position,spin(6.),240)[0])

class PileTests(unittest.TestCase):
    def test_authored_pile_matches_inventory_and_bag(self):
        rows=bag_pile.table()
        self.assertEqual(sorted(rows),list(range(24,144)))
        hx,hy=bag_pile.HALF[0]-bag_pile.WALL_MARGIN,bag_pile.HALF[1]-bag_pile.WALL_MARGIN
        for index,row in rows.items():
            self.assertEqual(row['kind'],index%3)
            d,h,_=bag_pile.CAN_SPECS[row['kind']];x,y,z=row['centre_m'];psi=row['axis_yaw_rad']
            a,b=h/2+bag_pile.CLEARANCE,d/2+bag_pile.CLEARANCE
            self.assertLessEqual(abs(x-bag_pile.CENTRE[0])+a*abs(np.cos(psi))+b*abs(np.sin(psi)),hx+1e-9)
            self.assertLessEqual(abs(y-bag_pile.CENTRE[1])+a*abs(np.sin(psi))+b*abs(np.cos(psi)),hy+1e-9)
            self.assertGreaterEqual(z-d/2,bag_pile.FLOOR_Z-1e-9)
            self.assertLess(row['top_m'],.80)
        position,rotation=bag_pile.pose(30,.4)
        self.assertAlmostEqual(abs(rotation.apply([0,0,1])[2]),0.,places=12)

if __name__=='__main__':unittest.main()
