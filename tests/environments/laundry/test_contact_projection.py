"""Targeted algebra checks; complete garment trials remain the physical evidence."""
import numpy as np
import pytest
ipctk=pytest.importorskip("ipctk")
from laundry.newton_style3d_contact_projection import project
from laundry.newton_style3d_contact_projection import ccd_witness_collides


def test_normal_projection_preserves_free_motion_and_mass_weighted_translation():
    base=np.array([[0.,0.,0.],[.03,0.,0.],[0.,.03,0.]])
    origin=np.vstack((base,base+[.002,.002,.001],[[1.,1.,1.]]))
    faces=np.array([[0,1,2],[3,4,5]],np.int32)
    mesh=ipctk.CollisionMesh(origin,ipctk.edges(faces),faces)
    collisions=ipctk.NormalCollisions(); collisions.build(mesh,origin,.004,0.)
    mass=np.array([1.,1.,1.,2.,2.,2.,1.])
    proposed=origin.copy(); proposed[3:6,2]-=.002; proposed[6]+=[.01,.02,.03]
    revised,stats=project(mesh,collisions,origin,proposed,1/mass)
    assert stats["corrected_vertices"]>0
    np.testing.assert_array_equal(revised[6],proposed[6])
    np.testing.assert_allclose(np.sum(mass[:,None]*(revised-proposed),axis=0),0.,atol=1e-14)
    np.testing.assert_allclose(revised[:6,:2],proposed[:6,:2],atol=1e-14)
    assert ipctk.compute_collision_free_stepsize(mesh,origin,revised)>.99


def test_cloth_correction_also_respects_existing_table_support():
    base=np.array([[0.,0.,.001],[.03,0.,.001],[0.,.03,.001]])
    origin=np.vstack((base,base+[.002,.002,.001]))
    faces=np.array([[0,1,2],[3,4,5]],np.int32)
    mesh=ipctk.CollisionMesh(origin,ipctk.edges(faces),faces)
    collisions=ipctk.NormalCollisions(); collisions.build(mesh,origin,.004,0.)
    proposed=origin.copy(); proposed[3:6,2]-=.002
    table=dict(half_x=1.,half_y=1.,top=0.,bottom=-.1,radius=np.full(6,.0005),epsilon=1e-6)
    revised,stats=project(mesh,collisions,origin,proposed,np.ones(6),table=table)
    # Iterative residual is ~1.5 pm here, far below float32 storage precision.
    assert np.min(revised[:,2]) >= .000501-1e-10
    assert stats['table_corrected_vertices']>0
    assert ipctk.compute_collision_free_stepsize(mesh,origin,revised)>.99


def test_cached_face_vertex_query_keeps_point_face_order_and_checks_motion():
    origin=np.array([[0.,0.,0.],[.03,0.,0.],[0.,.03,0.],[.004,.004,.001]])
    faces=np.array([[0,1,2]],np.int32)
    mesh=ipctk.CollisionMesh(origin,ipctk.edges(faces),faces)
    witness=dict(kind="FaceVertexCandidate",ids=[0,3])
    crossing=origin.copy(); crossing[3,2]=-.001
    assert ccd_witness_collides(mesh,origin,crossing,witness)
    assert not ccd_witness_collides(mesh,origin,origin,witness)
    assert not ccd_witness_collides(mesh,origin,crossing,None)


def test_cached_edge_query_does_not_certify_a_changed_geometry_without_ccd():
    origin=np.array([[-.01,0.,0.],[.01,0.,0.],[0.,-.01,.001],[0.,.01,.001]])
    edges=np.array([[0,1],[2,3]],np.int32)
    mesh=ipctk.CollisionMesh(origin,edges,np.zeros((0,3),np.int32))
    witness=dict(kind="EdgeEdgeCandidate",ids=[0,1])
    crossing=origin.copy(); crossing[2:,2]=-.001
    assert ccd_witness_collides(mesh,origin,crossing,witness)
    assert not ccd_witness_collides(mesh,origin,origin,witness)
