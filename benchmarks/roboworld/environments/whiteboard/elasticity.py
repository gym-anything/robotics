"""Resolved, body-local tetrahedral elasticity for bonded tool material.

P1/P2 displacement elements, isotropic small-strain elasticity, exact Dirichlet
elimination, and unilateral nodal contact. The solid equilibrates quasi-statically
within a Newton rigid-body step; its mass remains in the authored tool inertia.
This is not an independent time-stepped particle engine or a hand attachment.
Validity and inertia/error checks are part of the component audit.
"""

from collections import Counter, OrderedDict
from functools import lru_cache
import itertools
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def lattice(points, cells, periodic=False):
    nx, ny, nz = cells
    index = np.arange(len(points)).reshape(nx if periodic else nx + 1, ny + 1, nz + 1)
    tetra = []
    for x, y, z in itertools.product(range(nx), range(ny), range(nz)):
        corners = [
            index[(x + a) % index.shape[0], y + b, z + c]
            for a, b, c in itertools.product((0, 1), repeat=3)
        ]
        # Six consistently oriented simplices around the same cube diagonal.
        for order in itertools.permutations((0, 1, 2)):
            p = [0, 0, 0]
            path = [corners[0]]
            for axis in order:
                p[axis] = 1
                path.append(corners[p[0] * 4 + p[1] * 2 + p[2]])
            tetra.append(path)
    return np.asarray(points, float), np.asarray(tetra, np.int32)


def box_mesh(size, cells, center=(0, 0, 0)):
    axes = [np.linspace(-s / 2, s / 2, n + 1) + c for s, n, c in zip(size, cells, center)]
    return lattice(list(itertools.product(*axes)), cells)


def cap_mesh(around=24, layers=4, thickness=1):
    # A clearance bore with three integral retention ribs. Rib section and bore
    # are explicit engineering geometry, not a claimed product CT measurement.
    points = []
    for i in range(around):
        angle = i * 2 * np.pi / around
        distance = min(
            abs((angle - a + np.pi) % (2 * np.pi) - np.pi)
            for a in (0, 2 * np.pi / 3, 4 * np.pi / 3)
        )
        inner = 0.00755 - 0.00060 * max(0, 1 - distance / (np.pi / 12))
        for r in np.linspace(inner, 0.0085, thickness + 1):
            for z in np.linspace(-0.0175, 0.0161, layers + 1):
                points.append((r * np.cos(angle), r * np.sin(angle), z))
    return lattice(points, (around, thickness, layers), periodic=True)


class Solid:
    def __init__(self, points, tetra, fixed, young, poisson, *, quadratic=False):
        self.rest = np.asarray(points, float)
        self.tetra = np.asarray(tetra, np.int32).copy()
        self.fixed = np.asarray(fixed, bool)
        if self.fixed.shape != (len(points),) or not self.fixed.any():
            raise ValueError("A bonded root is required")
        if not young > 0 or not -1 < poisson < 0.5:
            raise ValueError("Invalid isotropic elastic constants")
        self.quadratic = quadratic
        for i, t in enumerate(self.tetra):
            if np.linalg.det((self.rest[t[1:]] - self.rest[t[0]]).T) < 0:
                self.tetra[i, [1, 2]] = self.tetra[i, [2, 1]]
        edges = {}
        edge_order = list(itertools.combinations(range(4), 2))
        if quadratic:
            points = self.rest.tolist()
            fixed = self.fixed.tolist()
            elements = []
            for t in self.tetra:
                mids = []
                for a, b in edge_order:
                    key = tuple(sorted((int(t[a]), int(t[b]))))
                    if key not in edges:
                        edges[key] = len(points)
                        points.append(((self.rest[key[0]] + self.rest[key[1]]) / 2).tolist())
                        fixed.append(bool(self.fixed[key[0]] and self.fixed[key[1]]))
                    mids.append(edges[key])
                elements.append([*t, *mids])
            self.rest = np.asarray(points, float)
            self.fixed = np.asarray(fixed, bool)
            self.tetra = np.asarray(elements, int)
        self.young, self.poisson = young, poisson
        mu = young / (2 * (1 + poisson))
        lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        d = np.zeros((6, 6))
        d[:3, :3] = lam
        d[np.arange(3), np.arange(3)] += 2 * mu
        d[3:, 3:] = np.eye(3) * mu
        self.constitutive = d
        nd = self.rest.size
        element_dofs = 30 if quadratic else 12
        entries = len(self.tetra) * element_dofs**2
        # Typed triplets avoid tens of millions of boxed Python integers and
        # floats during fine-mesh assembly. Same element/COO insertion order.
        rows = np.empty(entries, np.int32)
        cols = np.empty(entries, np.int32)
        values = np.empty(entries, np.float64)
        self.B = np.empty((len(self.tetra), 4 if quadratic else 1, 6, element_dofs))
        self.volumes = np.empty(len(self.tetra))
        for i, t in enumerate(self.tetra):
            x = self.rest[t[:4]]
            determinant = np.linalg.det((x[1:] - x[0]).T)
            if abs(determinant) < 1e-18:
                raise ValueError("Degenerate tetrahedron")
            if determinant < 0:
                self.tetra[i, [1, 2]] = self.tetra[i, [2, 1]]
                t = self.tetra[i]
                x = self.rest[t]
            volume = abs(determinant) / 6
            gradient = np.linalg.inv(np.c_[np.ones(4), x])[1:, :].T
            quadrature = []
            if quadratic:
                for g in range(4):
                    bary = np.full(4, (5 - np.sqrt(5)) / 20)
                    bary[g] = (5 + 3 * np.sqrt(5)) / 20
                    derivatives = [(4 * bary[j] - 1) * gradient[j] for j in range(4)]
                    derivatives.extend(
                        4 * (bary[a] * gradient[b] + bary[b] * gradient[a]) for a, b in edge_order
                    )
                    quadrature.append(derivatives)
            else:
                quadrature.append(gradient)
            matrices = []
            for derivatives in quadrature:
                B = np.zeros((6, 3 * len(t)))
                for j, (dx, dy, dz) in enumerate(derivatives):
                    B[:, 3 * j : 3 * j + 3] = [
                        [dx, 0, 0],
                        [0, dy, 0],
                        [0, 0, dz],
                        [dy, dx, 0],
                        [0, dz, dy],
                        [dz, 0, dx],
                    ]
                matrices.append(B)
            ids = (3 * t[:, None] + np.arange(3)).ravel()
            size = len(ids)
            ke = sum(B.T @ d @ B for B in matrices) * volume / len(matrices)
            section = slice(i * size**2, (i + 1) * size**2)
            rows[section] = np.repeat(ids, size)
            cols[section] = np.tile(ids, size)
            values[section] = ke.ravel()
            self.B[i] = matrices
            self.volumes[i] = volume
        self.K = coo_matrix((values, (rows, cols)), shape=(nd, nd)).tocsr()
        # The variational Hessian is symmetric. Remove only independent
        # floating-point summation order differences before factorization.
        self.K = (self.K + self.K.T) * 0.5
        self.free = np.flatnonzero(np.repeat(~self.fixed, 3))
        self._factor = splu(
            self.K[self.free][:, self.free].tocsc(),
            permc_spec="MMD_AT_PLUS_A",
            options={"SymmetricMode": True},
        )
        self._contact_blocks = OrderedDict()
        self._contact_cache_bytes = 0
        self._contact_cache_budget = 256 * 1024**2
        self._free_index = np.full(nd, -1, int)
        self._free_index[self.free] = np.arange(len(self.free))
        faces = []
        for t in self.tetra:
            a, b, c, d = t[:4]
            faces.extend(((a, c, b), (a, b, d), (a, d, c), (b, c, d)))
        counts = Counter(tuple(sorted(f)) for f in faces)
        faces = [f for f in faces if counts[tuple(sorted(f))] == 1]
        if quadratic:
            split = []
            for a, b, c in faces:
                ab = edges[tuple(sorted((a, b)))]
                bc = edges[tuple(sorted((b, c)))]
                ca = edges[tuple(sorted((c, a)))]
                split.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
            faces = split
        self.faces = np.asarray(faces, np.int32)
        self._contact_universe = np.unique(self.faces)
        self._contact_universe = self._contact_universe[~self.fixed[self._contact_universe]]
        self._set_contact_rows()

    def _set_contact_rows(self):
        self._contact_row_index = np.full(len(self.rest), -1, int)
        self._contact_row_index[self._contact_universe] = np.arange(len(self._contact_universe))
        self._contact_rows = self._free_index[
            (3 * self._contact_universe[:, None] + np.arange(3)).ravel()
        ]

    def displacement(self, force):
        u = np.zeros(self.rest.size)
        u[self.free] = self._factor.solve(np.asarray(force).reshape(-1)[self.free])
        return u.reshape(-1, 3)

    def energy(self, u):
        vector = np.asarray(u).ravel()
        return float(0.5 * vector @ self.K @ vector)

    def strain(self, u):
        strain = np.einsum(
            "tgij,tj->tgi", self.B, np.asarray(u)[self.tetra].reshape(len(self.tetra), -1)
        )
        return strain[:, 0] if not self.quadratic else strain

    @staticmethod
    def _normal_columns(blocks, node_ids, rows, left_normals, right_vectors, *, packed_cache=None):
        """Exact condensed columns with bounded, vectorized contractions.

        Chunking changes only temporary array construction, not contact sets,
        quadrature, the KKT test or reduction order within a 3-vector dot.
        """
        # Only immutable K^-1 blocks are cached. Normals, load vectors, active
        # selection and KKT gaps are recomputed for every physical contact.
        budget = 16 * 1024**2
        key = (tuple(node_ids), tuple(rows))
        packed = None if packed_cache is None else packed_cache.get(key)
        if packed is None and len(rows) * len(node_ids) * 9 * 8 <= budget:
            packed = np.stack([blocks[int(node)][rows] for node in node_ids])
            if packed_cache is not None:
                packed_cache[key] = packed
                while sum(p.nbytes for p in packed_cache.values()) > budget:
                    packed_cache.popitem(last=False)
        if packed is not None:
            if packed_cache is not None and key in packed_cache:
                packed_cache.move_to_end(key)
            vectors = np.einsum("jicd,jd->jic", packed, right_vectors)
            return np.einsum("ic,jic->ij", left_normals, vectors)
        result = np.empty((len(rows), len(node_ids)))
        for start in range(0, len(node_ids), 16):
            selected = node_ids[start : start + 16]
            packed = np.stack([blocks[int(node)][rows] for node in selected])
            vectors = np.einsum("jicd,jd->jic", packed, right_vectors[start : start + 16])
            result[:, start : start + 16] = np.einsum("ic,jic->ij", left_normals, vectors)
        return result

    def contact(
        self,
        nodes,
        normals,
        penetration,
        *,
        tolerance=2e-9,
        external_force=None,
        mobility=None,
        resolve_displacement=True,
        warm_start_nodes=None,
    ):
        """Minimize elastic energy subject to n·u >= penetration, lambda >= 0.

        Primal/dual KKT residuals are checked, never silently accepted at an
        iteration cap. Every active contact has a three-dimensional deformation
        field and a reciprocal force; inactive material also deforms through K.
        """
        nodes = np.asarray(nodes, int)
        if not resolve_displacement and external_force is not None:
            raise ValueError("Lazy reconstruction requires explicitly stored contact loads only")

        def mobility_block(ids):
            if isinstance(mobility, tuple):
                jac, mass_inverse = mobility
                selected = jac[ids]
                return selected @ mass_inverse @ selected.T
            return np.asarray(mobility)[np.ix_(ids, ids)]

        def mobility_apply(load):
            if isinstance(mobility, tuple):
                jac, mass_inverse = mobility
                return jac @ (mass_inverse @ (jac.T @ load))
            return np.asarray(mobility) @ load

        normals = np.asarray(normals, float)
        p = np.asarray(penetration, float)
        background = (
            np.zeros_like(self.rest)
            if external_force is None
            else self.displacement(external_force)
        )
        p = p - np.einsum("ij,ij->i", background[nodes], normals)
        if len(nodes) == 0 or np.max(p) <= 0:
            return background, np.zeros((len(nodes), 3)), {"iterations": 0, "residual_m": 0.0}
        if np.any(self.fixed[nodes]):
            raise ValueError("Contact cannot prescribe penetration at a bonded root")
        if np.any(self._contact_row_index[nodes] < 0):
            self._contact_universe = np.union1d(self._contact_universe, nodes)
            self._set_contact_rows()
            self._contact_blocks.clear()
            if hasattr(self, "_packed_contact_blocks"):
                self._packed_contact_blocks.clear()
            self._contact_cache_bytes = 0
        contact_rows = self._contact_row_index[nodes]
        # A previous support is an optimization only. Every inactive candidate
        # still receives the same KKT gap check before a result is accepted.
        active = set(np.flatnonzero(p > 0).tolist())
        if warm_start_nodes is not None:
            previous = set(np.flatnonzero(np.isin(nodes, warm_start_nodes)).tolist())
            if previous:
                active = previous
        load = np.zeros(len(nodes))
        seen = set()
        for iteration in range(4 * len(nodes) + 10):
            signature = tuple(sorted(active))
            single_pivot = signature in seen
            seen.add(signature)
            ids = sorted(active)
            load[:] = 0
            projected = np.zeros((0, 0))
            if ids:
                active_nodes = nodes[ids]
                active_normals = normals[ids]
                # K^-1 columns depend on the material node, NOT on the active
                # contact set or changing normals. Cache each node once rather
                # than duplicating the same columns for thousands of sets.
                missing = [
                    int(node) for node in active_nodes if int(node) not in self._contact_blocks
                ]
                for start in range(0, len(missing), 16):
                    batch = missing[start : start + 16]
                    dofs = self._free_index[(3 * np.asarray(batch)[:, None] + np.arange(3)).ravel()]
                    rhs = np.zeros((len(self.free), len(dofs)))
                    rhs[dofs, np.arange(len(dofs))] = 1
                    solved = self._factor.solve(rhs)
                    for index, node in enumerate(batch):
                        block = (
                            solved[self._contact_rows, 3 * index : 3 * index + 3]
                            .reshape(-1, 3, 3)
                            .copy()
                        )
                        self._contact_blocks[node] = block
                        self._contact_cache_bytes += block.nbytes
                # Exact static condensation onto contact coordinates. We need
                # only normal displacements during pivots, not a concatenated
                # n_free x 3*n_contact matrix of full interior displacements.
                if not hasattr(self, "_packed_contact_blocks"):
                    self._packed_contact_blocks = OrderedDict()
                projected = self._normal_columns(
                    self._contact_blocks,
                    active_nodes,
                    contact_rows[ids],
                    active_normals,
                    active_normals,
                    packed_cache=self._packed_contact_blocks,
                )
                # Bounded reusable cache. Current solve owns its columns even
                # if an exceptionally broad contact set exceeds this budget.
                for node in active_nodes:
                    self._contact_blocks.move_to_end(int(node))
                current_blocks = {
                    int(node): self._contact_blocks[int(node)] for node in active_nodes
                }
                while self._contact_cache_bytes > self._contact_cache_budget:
                    _, old = self._contact_blocks.popitem(last=False)
                    self._contact_cache_bytes -= old.nbytes
                compliance = projected
                if mobility is not None:
                    compliance = compliance + mobility_block(ids)
                # Solve the convex dual QP with nonnegative loads directly.
                # Removing one negative unconstrained load at a time repeatedly
                # solved the full interior and could cycle between broad faces.
                # C=L L^T; nnls(L^T, L^-1 p) minimizes .5*l^T C*l-p^T*l.
                factor = np.linalg.cholesky((compliance + compliance.T) * 0.5)
                rhs = solve_triangular(factor, p[ids], lower=True)
                load[ids], _ = nnls(factor.T, rhs, maxiter=max(100, 10 * len(ids)))
                active = {i for i in ids if load[i] > 0}
                ids = sorted(active)
            # Check every candidate, including previously inactive nodes. The
            # lazy cap path evaluates the exact condensed field without an
            # unnecessary interior solve; the full path reconstructs it here.
            force = normals * np.maximum(load, 0)[:, None]
            if resolve_displacement:
                material_force = np.zeros_like(self.rest)
                np.add.at(material_force, nodes, force)
                u = self.displacement(material_force)
                normal_displacement = np.einsum("ij,ij->i", u[nodes], normals)
            else:
                if ids:
                    columns = self._normal_columns(
                        current_blocks,
                        nodes[ids],
                        contact_rows,
                        normals,
                        force[ids],
                        packed_cache=self._packed_contact_blocks,
                    )
                    # Preserve the previous left-to-right accumulation, rather
                    # than replacing it with a differently ordered BLAS sum.
                    normal_displacement = np.add.accumulate(columns, axis=1)[:, -1]
                else:
                    normal_displacement = np.zeros(len(nodes))
            gap = normal_displacement - p
            if mobility is not None:
                gap += mobility_apply(load)
            inactive = np.ones(len(nodes), dtype=bool)
            inactive[list(active)] = False
            violated = np.flatnonzero(inactive & (gap < -tolerance)).tolist()
            if violated:
                active.update([min(violated, key=lambda i: gap[i])] if single_pivot else violated)
                continue
            residual = max(float(max(0, -gap.min())), float(np.max(np.abs(gap[ids]))) if ids else 0)
            if residual > tolerance or load.min() < -1e-10:
                raise FloatingPointError("Elastic contact KKT residual exceeded tolerance")
            quality = {
                "iterations": iteration + 1,
                "residual_m": residual,
                "elastic_energy_j": float(0.5 * load @ normal_displacement),
            }
            if not resolve_displacement:
                # K is constant and quasi-static: (node ids, local forces, K)
                # fully determines u. Do not solve and serialize tens of
                # thousands of redundant interior values every physics tick.
                return None, force, quality
            reconstructed = np.einsum("ij,ij->i", u[nodes], normals) - p
            if mobility is not None:
                reconstructed += mobility_apply(load)
            residual = max(
                residual,
                float(max(0, -reconstructed.min())),
                float(np.max(np.abs(reconstructed[ids]))) if ids else 0,
            )
            if residual > tolerance:
                raise FloatingPointError("Condensed/full elastic reconstruction mismatch")
            quality["residual_m"] = residual
            return background + u, force, quality
        raise FloatingPointError("Elastic contact active set did not converge")

    def metadata(self):
        return {
            "formulation": f"P{2 if self.quadratic else 1} tetrahedra, isotropic small strain, quasi-static internal deformation",
            "vertices": len(self.rest),
            "tetrahedra": len(self.tetra),
            "young_pa": self.young,
            "poisson_ratio": self.poisson,
            "volume_m3": float(self.volumes.sum()),
            "root_nodes": np.flatnonzero(self.fixed).tolist(),
        }


def shore_a_modulus(hardness):
    """Gent's empirical Shore A hardness to Young's modulus relation, Pa."""
    return 0.0981 * (56 + 7.62336 * hardness) / (0.137505 * (254 - 2.54 * hardness)) * 1e6


# Bonded synthetic-fiber marker tip. QB/T 4164-2026 ("Fiber pen tip") bands hard
# tips by Shore A: low 35-55, medium 55-75, high >75 HA. The centre of the bounded
# bands, 55 HA, by Gent's relation: 2.97 MPa. A class prior, not a STAEDTLER value.
NIB_YOUNG_PA = shore_a_modulus(55)
# Soft wool felt: 3 psi compresses McMaster-Carr grade F13 by 10%, a 207 kPa
# secant modulus. A published felt grade, not a measured duster pad.
FELT_YOUNG_PA = 3 * 6894.757 / 0.10


@lru_cache(maxsize=12)
def tool_solid(kind, refinement=None):
    if refinement is not None and (isinstance(refinement, bool) or refinement not in (1, 2, 3, 4)):
        raise ValueError("refinement must be 1, 2, 3 or 4")
    if kind == "cap" and refinement is None:
        # Axial/radial resolution check: this mesh differs 1.7452% in insertion
        # load from isotropic r4.
        points, tets = cap_mesh(72, 12, 2)
        solid = Solid(points, tets, np.isclose(points[:, 2], 0.0161), 1.3e9, 0.42, quadratic=True)
        # Only the inner cap/barrel interface uses this constitutive solver;
        # outer shell contacts remain native rigid collision proxies. Compact
        # the exact Schur rows to that domain and retain its reusable columns.
        nodes = solid._contact_universe
        solid._contact_universe = nodes[np.linalg.norm(solid.rest[nodes, :2], axis=1) < 0.008]
        solid._set_contact_rows()
        solid._contact_cache_budget = 1536 * 1024**2
        return solid
    refinement = 1 if refinement is None else refinement
    if kind == "nib":
        points, tets = box_mesh(
            (0.005, 0.002, 0.0075), (2 * refinement, refinement, 3 * refinement), (0, 0, 0.05925)
        )
        return Solid(points, tets, np.isclose(points[:, 2], 0.0555), NIB_YOUNG_PA, 0.10)
    if kind == "felt":
        points, tets = box_mesh(
            (0.103, 0.053, 0.005), (6 * refinement, 4 * refinement, 2 * refinement), (0, 0, -0.0075)
        )
        return Solid(points, tets, np.isclose(points[:, 2], -0.005), FELT_YOUNG_PA, 0.10)
    if kind == "cap":
        points, tets = cap_mesh(24 * refinement, 4 * refinement, refinement)
        return Solid(points, tets, np.isclose(points[:, 2], 0.0161), 1.3e9, 0.42, quadratic=True)
    raise ValueError("Unknown elastic tool material")
