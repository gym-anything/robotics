"""Storage/filter checks only; no simulation is advanced."""
from types import SimpleNamespace
import unittest
import numpy as np
import warp as wp
from deposit_cans.topology import enable_bag_local_contact

class LocalTopologyTest(unittest.TestCase):
    def test_bag_only_filter_change_preserves_can_and_helper(self):
        edges=np.array([[2,3,0,1],[0,1,2,3],[6,7,4,5],[4,5,6,7]],np.int32)
        def array(value):return wp.array(np.asarray(value,np.int32),dtype=int,device='cpu')
        vertices=np.array([1,2,1,2,0,1,0,1,9,10,11,12,8,9,10,11,12,13],np.int32)
        offsets=np.arange(0,19,2,dtype=np.int32)
        d=SimpleNamespace(device='cpu',model=SimpleNamespace(particle_count=9,edge_indices=array(edges)),
            vertex_triangle_filtering_list=array(vertices),vertex_triangle_filtering_list_offsets=array(offsets),
            edge_filtering_list=array([1,0,3,2]),edge_filtering_list_offsets=array([0,1,2,3,4]))
        before=[vertices[offsets[i]:offsets[i+1]].copy()for i in range(9)]
        result=enable_bag_local_contact(d,np.arange(4))
        values=d.vertex_triangle_filtering_list.numpy();off=d.vertex_triangle_filtering_list_offsets.numpy()
        for i in range(9):
            row=values[off[i]:off[i+1]]
            if i<4:self.assertEqual(len(row),0)
            else:np.testing.assert_array_equal(row,before[i])
        np.testing.assert_array_equal(d.edge_filtering_list.numpy(),[3,2])
        np.testing.assert_array_equal(d.edge_filtering_list_offsets.numpy(),[0,0,0,1,2])
        self.assertEqual(result['vertex_triangle_filtering_list']['after'],10)

if __name__=='__main__':unittest.main()
