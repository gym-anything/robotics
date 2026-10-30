"""Development comparison: variational area on the particle alpha boundary.

No remeshing of material points and no volume/position correction. The
triangulation supplies quadrature only; Young's solid/liquid area replaces
liquid/air area on triangles touching the planar diagnostic wall.
"""

import numpy as np
from scipy.spatial import Delaunay
from .capillary import CapillaryPredictor


def alpha_boundary(q, radius):
    q = np.asarray(q, dtype=float)
    tet = Delaunay(q).simplices
    xyz = q[tet]
    a = 2*(xyz[:, 1:]-xyz[:, :1])
    rhs = (xyz[:, 1:]**2).sum(2)-(xyz[:, :1]**2).sum(2)
    determinant = np.linalg.det(a)
    valid = np.abs(determinant) > radius**3*1e-10
    centers = np.linalg.solve(a[valid], rhs[valid, :, None])[:, :, 0]
    keep = np.zeros(len(tet), dtype=bool)
    keep[valid] = np.linalg.norm(centers-xyz[valid, 0], axis=1) <= radius
    # Qhull can insert exactly flat tetrahedra on a regular particle lattice.
    # Dropping them before face cancellation creates fictitious interior area.
    edges = xyz[~valid, :, None]-xyz[~valid, None, :]
    keep[~valid] = np.linalg.norm(edges, axis=-1).max(axis=(1, 2)) <= 2*radius
    selected = tet[keep]
    faces = selected[:, np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])].reshape(-1, 3)
    centers = np.repeat(q[selected].mean(axis=1), 4, axis=0)
    tri = q[faces]
    reverse = np.einsum("ij,ij->i", np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), tri.mean(axis=1)-centers) < 0
    faces[reverse] = faces[reverse, ::-1]
    sorted_faces = np.sort(faces, axis=1)
    _, index, count = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
    return faces[index[count == 1]]


def area_energy_force(q, faces, coefficients):
    tri = q[faces]
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    normal = cross/np.maximum(length[:, None], 1e-30)
    f = np.zeros_like(q)
    for corner in range(3):
        edge = tri[:, (corner+1) % 3]-tri[:, (corner+2) % 3]
        gradient = .5*np.cross(edge, normal)
        np.add.at(f, faces[:, corner], -coefficients[:, None]*gradient)
    return float(.5*np.dot(coefficients, length)), f


class MeshCapillaryPredictor(CapillaryPredictor):
    def predict(self, state, dt):
        sigma = self.sigma_n_m
        self.sigma_n_m = 0.
        try:
            predicted = super().predict(state, dt)
        finally:
            self.sigma_n_m = sigma
        q = state.particle_q.numpy().astype(float)
        spacing = float(self.volumes.numpy()[0])**(1/3)
        self.faces = alpha_boundary(q, 2*spacing)
        coeff = np.full(len(self.faces), sigma)
        wall = getattr(self, "wall", None)
        gradient_coeff = np.zeros(len(self.faces))
        if wall is not None:
            # The liquid quadrature centres are not the material boundary.
            # Regularize the wall material transition over one particle spacing,
            # including the derivative of the spatial surface-energy coefficient.
            height = q[self.faces, 2].mean(axis=1)
            tri = q[self.faces]
            normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            # Only the solid-facing boundary can be a liquid/solid interface.
            # A thin sheet still has a positive-energy upper liquid/air face.
            facing = normals[:, 2] < -.5*np.linalg.norm(normals, axis=1)
            s = np.clip((height-spacing/2)/spacing, 0., 1.)
            wet = 1.-3*s*s+2*s*s*s
            contrast = sigma*(1.+np.cos(np.deg2rad(wall.angle_deg)))
            coeff -= contrast*wet*facing
            gradient_coeff = contrast*6*s*(1.-s)/spacing*facing
        self.energy_j, f = area_energy_force(q, self.faces, coeff)
        tri = q[self.faces]
        area = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
        for corner in range(3):
            np.add.at(f[:, 2], self.faces[:, corner], -area*gradient_coeff/3)
        self.forces.assign(f.astype(np.float32))
        predicted.particle_qd.assign((state.particle_qd.numpy()+dt*f/self.model.particle_mass.numpy()[:, None]).astype(np.float32))
        return predicted
