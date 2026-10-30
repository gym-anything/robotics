"""Body-attached volume quadrature of the actual Newton collision solids.

Used by the repository SPH extension, never as replacement rigid geometry.
Voxel centres are retained only inside the union of each fixed-connected
component's particle-collision shapes, retaining each node's actual owner.
Hollow mesh cavities therefore remain empty. Sampling is
restricted to a guarded region near the fluid; expansion only changes boundary
quadrature, never fluid/rigid state. Unsupported shape types fail explicitly.
"""

from dataclasses import dataclass
import itertools
import numpy as np
import warp as wp
from .runtime import require_newton


@wp.kernel
def inside_shape(points: wp.array[wp.vec3], transform: wp.transform, kind: int,
                 scale: wp.vec3, mesh: wp.uint64, closed: int, result: wp.array[int]):
    i = wp.tid()
    p = wp.transform_point(wp.transform_inverse(transform), points[i])
    d = float(1.e10)
    # Newton GeoType values are passed from the host as this module's small
    # explicit dispatch enum; no assumptions about library enum numbering.
    if kind == 0:
        v = wp.abs(p)-scale
        d = wp.length(wp.max(v, wp.vec3(0.)))+wp.min(wp.max(v[0], wp.max(v[1], v[2])), 0.)
    elif kind == 1:
        d = wp.length(p)-scale[0]
    elif kind == 2:
        p[2] -= wp.clamp(p[2], -scale[1], scale[1])
        d = wp.length(p)-scale[0]
    elif kind == 3:
        radial = wp.sqrt(p[0]*p[0]+p[1]*p[1])-scale[0]
        axial = wp.abs(p[2])-scale[1]
        rp, ap = wp.max(radial, 0.), wp.max(axial, 0.)
        d = wp.sqrt(rp*rp+ap*ap)+wp.min(wp.max(radial, axial), 0.)
    elif kind == 4:
        if closed != 0:
            query = wp.mesh_query_point_sign_winding_number(mesh, wp.cw_div(p, scale), 1.e6)
        else:
            query = wp.mesh_query_point_sign_normal(mesh, wp.cw_div(p, scale), 1.e6)
        if query.result:
            closest = wp.cw_mul(wp.mesh_eval_position(mesh, query.face, query.u, query.v), scale)
            d = query.sign*wp.length(p-closest)
    if d < 0.:
        result[i] = 1


@wp.kernel
def assign_owner(inside: wp.array[int], shape: int, owners: wp.array[int]):
    i = wp.tid()
    if inside[i] != 0:
        owners[i] = shape


@wp.kernel
def world_samples(local: wp.array[wp.vec3], ids: wp.array[int],
                  body_q: wp.array[wp.transform], body_v: wp.array[wp.spatial_vector], com: wp.array[wp.vec3],
                  q: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    i = wp.tid()
    b = ids[i]
    point, velocity = wp.vec3(local[i]), wp.vec3(0.)
    if b >= 0:
        point = wp.transform_point(body_q[b], point)
        offset = point-wp.transform_point(body_q[b], com[b])
        velocity = wp.spatial_top(body_v[b])+wp.cross(wp.spatial_bottom(body_v[b]), offset)
    q[i] = point
    v[i] = velocity


def transformed(points, pose):
    from scipy.spatial.transform import Rotation
    pose = np.asarray(pose)
    return np.asarray(points)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]


def inverse_transformed(points, pose):
    from scipy.spatial.transform import Rotation
    pose = np.asarray(pose)
    return (np.asarray(points)-pose[:3])@Rotation.from_quat(pose[3:]).as_matrix()


def corners(low, high):
    return np.array(list(itertools.product(*zip(low, high))))


@dataclass
class Shape:
    index: int
    kind: int
    pose: object
    scale: object
    mesh: object
    bounds: np.ndarray
    closed: int


class SolidSamples:
    def __init__(self, builder, model, spacing, *, margin=.01, maximum_points=2_000_000,
                 shape_contact_angles=None, coverage_support_m=None, particle_regions=None):
        newton, _ = require_newton()
        if spacing <= 0 or margin <= 6*spacing:
            raise ValueError("Positive boundary spacing and guard exceeding two surface supports required")
        self.model, self.spacing, self.margin = model, float(spacing), float(margin)
        self.coverage_support_m = float(coverage_support_m or 3*spacing)
        if not np.isfinite(self.coverage_support_m) or self.coverage_support_m <= 0 or margin <= 2*self.coverage_support_m:
            raise ValueError('Boundary guard must exceed twice the fluid surface support')
        self.maximum_points = int(maximum_points)
        # Fixed parcel-index groups are ONLY a conservative spatial query
        # partition. Parcels may leave their original vessel and mix normally;
        # every current position participates in its group's bounds each step.
        self.particle_regions = None if particle_regions is None else tuple(particle_regions)
        self.region_count = len(self.particle_regions) if self.particle_regions is not None else 1
        if self.region_count < 1:
            raise ValueError('Boundary query regions must cover the liquid')
        self.shape_contact_angles = shape_contact_angles
        self.sample_contact_angles = None
        self.groups = {}
        self.shape_owner = {}
        self.coverage = {}
        self.meshes = []
        self.rebuilds = 0
        kinds = {"BOX": 0, "SPHERE": 1, "CAPSULE": 2, "CYLINDER": 3, "MESH": 4, "CONVEX_MESH": 4}
        from newton._src.geometry.flags import MeshProperties
        properties = model._shape_mesh_properties.numpy()
        query_mesh_ids = np.zeros(model.shape_count, np.uint64)
        self.meridians = getattr(builder, "titration_meridians", {})
        self.analytic = getattr(builder, 'titration_analytic_solids', {})
        analytic_kinds = np.zeros(model.shape_count, np.int32)
        analytic_dimensions = np.zeros((model.shape_count, 4), np.float32)
        for shape, (kind, dimensions) in self.analytic.items():
            analytic_kinds[shape] = kind
            analytic_dimensions[shape] = dimensions
        self.analytic_kinds = wp.array(analytic_kinds, dtype=int, device=model.device)
        self.analytic_dimensions = wp.array(analytic_dimensions, dtype=wp.vec4, device=model.device)
        profile_data, layout = [], np.zeros((model.shape_count, 3), np.int32)
        for shape, (profile, segments) in self.meridians.items():
            layout[shape] = (len(profile_data), len(profile), segments)
            profile_data.extend(profile)
        self.profile_data = wp.array(np.asarray(profile_data, np.float32).reshape(-1, 2), dtype=wp.vec2, device=model.device)
        self.profile_layout = wp.array(layout, dtype=wp.vec3i, device=model.device)
        self.profile_layout_host = layout
        for index in range(builder.shape_count):
            if not builder.shape_flags[index] & int(newton.ShapeFlags.COLLIDE_PARTICLES):
                continue
            name = newton.GeoType(builder.shape_type[index]).name
            if name not in kinds:
                raise ValueError(f"Boundary-volume sampling not implemented for {name}: {builder.shape_label[index]}")
            kind, pose = kinds[name], builder.shape_transform[index]
            scale = np.asarray(builder.shape_scale[index], dtype=float)
            source, mesh = builder.shape_source[index], None
            closed = int(bool(properties[index] & int(MeshProperties.WATERTIGHT)))
            if kind == 4:
                vertices = np.asarray(source.vertices, np.float32)
                mesh = wp.Mesh(wp.array(vertices, dtype=wp.vec3, device=model.device),
                    wp.array(np.asarray(source.indices).reshape(-1), dtype=int, device=model.device),
                    support_winding_number=bool(closed))
                query_mesh_ids[index] = mesh.id
                self.meshes.append(mesh)
                positions = vertices*scale
                low, high = positions.min(0), positions.max(0)
            else:
                extent = scale.copy() if kind == 0 else np.full(3, scale[0])
                if kind in (2, 3):
                    extent[2] = scale[1]+(scale[0] if kind == 2 else 0.)
                low, high = -extent, extent
            points = transformed(corners(low, high), pose)
            shape = Shape(index, kind, pose, scale, mesh, np.array([points.min(0), points.max(0)]), closed)
            self.groups.setdefault(builder.shape_body[index], []).append(shape)
            self.shape_owner[index] = builder.shape_body[index]
        self.owner_bodies = set(self.shape_owner.values())
        self.original_groups = self.groups
        # Fixed-connected collision pieces form one solid-volume union. A
        # socket clamped to world must not add a second volume where it meets
        # a world-static column. Keep actual owners for reaction attribution.
        anchors = {body: body for body in range(-1, model.body_count)}
        def root(body):
            while anchors[body] != body:
                anchors[body] = anchors[anchors[body]]
                body = anchors[body]
            return body
        for kind, parent, child, enabled in zip(builder.joint_type, builder.joint_parent,
                                                builder.joint_child, builder.joint_enabled):
            if enabled and kind == int(newton.JointType.FIXED):
                a, b = root(parent), root(child)
                anchors[max(a,b)] = min(a,b)
        self.anchor_for_body = {body: root(body) for body in anchors}
        initial = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, initial)
        poses = initial.body_q.numpy()
        merged = {}
        for owner, shapes in self.original_groups.items():
            anchor = self.anchor_for_body[owner]
            owner_pose = wp.transform(wp.vec3(*poses[owner,:3]), wp.quat(*poses[owner,3:])) if owner >= 0 else wp.transform_identity()
            anchor_pose = wp.transform(wp.vec3(*poses[anchor,:3]), wp.quat(*poses[anchor,3:])) if anchor >= 0 else wp.transform_identity()
            relative = wp.transform_multiply(wp.transform_inverse(anchor_pose), owner_pose)
            for s in shapes:
                if owner == anchor:
                    shape = s
                else:
                    bounds = transformed(corners(*s.bounds), relative)
                    shape = Shape(s.index, s.kind, wp.transform_multiply(relative,s.pose),
                        s.scale, s.mesh, np.array([bounds.min(0),bounds.max(0)]),s.closed)
                merged.setdefault(anchor,[]).append(shape)
        self.groups = merged
        self.group_bounds = {body: np.array([np.min([s.bounds[0] for s in shapes], axis=0),
            np.max([s.bounds[1] for s in shapes], axis=0)]) for body, shapes in merged.items()}
        self.local = None
        self.mesh_query_ids = wp.array(query_mesh_ids, dtype=wp.uint64, device=model.device)
        self.fluid_bounds = None
        self.grid = wp.HashGrid(64, 64, 512, device=model.device)

    def update(self, body_state, fluid_positions):
        if self.particle_regions is not None:
            return self._update_regions(body_state, fluid_positions)
        # Transform the eight corners of the complete fluid AABB, not every
        # point once per rigid body. This is conservative (may add candidates,
        # cannot omit a point), and avoids O(particles*bodies) CPU reductions.
        if isinstance(fluid_positions, wp.array):
            from .particle_bounds import ParticleBounds
            from .boundary_coverage import BoundaryCoverage
            if self.fluid_bounds is None:
                self.fluid_bounds = ParticleBounds(len(fluid_positions), self.model.device)
            if not hasattr(self, 'coverage_check'):
                self.coverage_check = BoundaryCoverage(self.groups, self.model.device, self.group_bounds)
            bounds = self.fluid_bounds.on_device(fluid_positions)
            rebuild = self.coverage_check.needs_rebuild(bounds, body_state.body_q,
                self.coverage, self.coverage_support_m)
            if self.local is None or rebuild:
                self._rebuild(self.coverage_check.read(), body_state.body_q.numpy())
            wp.launch(world_samples, dim=len(self.local), inputs=[self.local, self.body_ids, body_state.body_q,
                body_state.body_qd, self.model.body_com], outputs=[self.q, self.v], device=self.model.device)
            self.grid.build(self.q, self.coverage_support_m)
            return
        else:
            q = np.asarray(fluid_positions)
            low, high = q.min(0), q.max(0)
        q = corners(low, high)
        poses = body_state.body_q.numpy()
        rebuild = self.local is None
        current = {}
        for body in self.groups:
            local_fluid = inverse_transformed(q, poses[body]) if body >= 0 else q
            bounds = np.array([local_fluid.min(0), local_fluid.max(0)])
            current[body] = bounds
            old = self.coverage.get(body)
            if old is None or np.any(bounds[0]-self.coverage_support_m < old[0]) or np.any(bounds[1]+self.coverage_support_m > old[1]):
                rebuild = True
        if rebuild:
            self._rebuild(current, poses)
        wp.launch(world_samples, dim=len(self.local), inputs=[self.local, self.body_ids, body_state.body_q,
            body_state.body_qd, self.model.body_com], outputs=[self.q, self.v], device=self.model.device)
        self.grid.build(self.q, self.coverage_support_m)

    def _update_regions(self, body_state, positions):
        from .particle_bounds import ParticleBounds
        from .boundary_coverage import BoundaryCoverage
        ranges = self.particle_regions
        if ranges[0][0] != 0 or ranges[-1][1] != len(positions) or any(
                type(a) is not int or type(b) is not int or b <= a or
                (i and a != ranges[i-1][1]) for i, (a, b) in enumerate(ranges)):
            raise ValueError('Boundary query regions must partition every parcel exactly once')
        if not hasattr(self, 'regional_bounds'):
            self.regional_bounds = [ParticleBounds(b-a, self.model.device) for a, b in ranges]
            self.regional_checks = [BoundaryCoverage(self.groups, self.model.device, self.group_bounds) for _ in ranges]
            self.regional_coverage_source = None
        if self.regional_coverage_source is not self.coverage:
            self.regional_coverages = [{body: np.asarray(value).reshape(-1, 2, 3)[i]
                for body, value in self.coverage.items()} for i in range(len(ranges))]
            self.regional_coverage_source = self.coverage
        rebuild = self.local is None
        values = []
        if isinstance(positions, wp.array):
            for i, (a, b) in enumerate(ranges):
                bounds = self.regional_bounds[i].on_device(positions[a:b])
                required = self.regional_checks[i].needs_rebuild(bounds, body_state.body_q,
                    self.regional_coverages[i], self.coverage_support_m)
                rebuild = rebuild or required
            if rebuild:
                values = [check.read() for check in self.regional_checks]
        else:
            poses = body_state.body_q.numpy()
            for i, (a, b) in enumerate(ranges):
                cloud = np.asarray(positions[a:b], dtype=np.float64)
                cube = corners(cloud.min(0), cloud.max(0))
                item = {}
                for body in self.groups:
                    local = inverse_transformed(cube, poses[body].astype(float)) if body >= 0 else cube
                    bounds = np.array([local.min(0), local.max(0)])
                    item[body] = bounds
                    old = self.regional_coverages[i].get(body)
                    if old is None or np.any(bounds[0]-self.coverage_support_m < old[0]) or np.any(bounds[1]+self.coverage_support_m > old[1]):
                        rebuild = True
                values.append(item)
        if rebuild:
            current = {body: np.stack([value[body] for value in values]) for body in self.groups}
            self._rebuild(current, body_state.body_q.numpy())
        wp.launch(world_samples, dim=len(self.local), inputs=[self.local, self.body_ids,
            body_state.body_q, body_state.body_qd, self.model.body_com], outputs=[self.q, self.v], device=self.model.device)
        self.grid.build(self.q, self.coverage_support_m)

    def _rebuild(self, current, poses):
        chunks, identities, candidates, angle_chunks = [], [], set(), []
        count = 0
        coverage = {}
        for body, shapes in self.groups.items():
            regions = np.asarray(current[body]).reshape(-1, 2, 3)+np.array([[[-self.margin], [self.margin]]])
            coverage[body] = regions[0] if self.particle_regions is None else regions
            point_chunks, owner_chunks = [], []
            for low, high in regions:
                overlap = [s for s in shapes if np.all(s.bounds[0] < high) and np.all(s.bounds[1] > low)]
                if not overlap:
                    continue
                candidates.update(s.index for s in overlap)
                for points, owner in self._sample_patch(low, high, overlap):
                    point_chunks.append(points)
                    owner_chunks.append(owner)
            if not point_chunks:
                continue
            points, owner = np.concatenate(point_chunks), np.concatenate(owner_chunks)
            if len(regions) > 1:
                # Regions may overlap. A body-local quadrature node represents
                # one volume, never one volume per overlapping search box.
                points, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
                owners = np.full(len(points), -1, np.int32)
                np.maximum.at(owners, inverse, owner)
                owner = owners
            owner_ids = np.asarray([self.shape_owner[int(i)] for i in owner], np.int32)
            if self.shape_contact_angles is not None:
                angle_chunks.append(np.asarray([self.shape_contact_angles[int(i)] for i in owner], float))
            for actual in np.unique(owner_ids):
                if actual != body:
                    mask = owner_ids == actual
                    world_points = transformed(points[mask], poses[body]) if body >= 0 else points[mask]
                    points[mask] = inverse_transformed(world_points, poses[actual]) if actual >= 0 else world_points
            chunks.append(points)
            identities.extend(owner_ids.tolist())
            count += len(points)
            if count > self.maximum_points:
                raise ValueError('Boundary quadrature budget exceeded; no samples discarded')
        points = np.concatenate(chunks) if chunks else np.empty((0, 3), np.float32)
        self.local = wp.array(points, dtype=wp.vec3, device=self.model.device)
        self.body_ids = wp.array(identities, dtype=int, device=self.model.device)
        self.q, self.v = wp.empty_like(self.local), wp.empty_like(self.local)
        self.coverage = coverage
        self.sample_contact_angles = np.concatenate(angle_chunks) if angle_chunks else None
        self.candidate_shape_ids = sorted(candidates)
        self.rebuilds += 1

    def _sample_patch(self, low, high, shapes):
        d = self.spacing
        low = np.maximum(low, np.min([s.bounds[0] for s in shapes], axis=0))
        high = np.minimum(high, np.max([s.bounds[1] for s in shapes], axis=0))
        start, stop = np.floor(low/d).astype(int), np.ceil(high/d).astype(int)
        # Dispensed parcels can span the air between the burette and bench.
        # This bounds total search work, not one allocation: sampling below
        # still uses eight-z-slice slabs and keeps the retained-node budget.
        candidates = int(np.prod(stop-start))
        if candidates > 100_000_000:
            raise ValueError(f'Boundary sampling candidate budget exceeded: {candidates} > 100000000; no resolution substitution')
        for z in range(start[2], stop[2], 8):
            axes = [(np.arange(start[k], stop[k])+.5)*d for k in (0, 1)]
            axes += [(np.arange(z, min(z+8, stop[2]))+.5)*d]
            points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3).astype(np.float32)
            gp = wp.array(points, dtype=wp.vec3, device=self.model.device)
            inside = wp.zeros(len(points), dtype=int, device=self.model.device)
            owners = wp.full(len(points), -1, dtype=int, device=self.model.device)
            for shape in shapes:
                inside.zero_()
                if shape.index in self.analytic:
                    from .analytic_solids import inside as inside_analytic
                    kind, dimensions = self.analytic[shape.index]
                    wp.launch(inside_analytic, dim=len(points), inputs=[gp, shape.pose,
                        kind, wp.vec4(*dimensions)], outputs=[inside], device=self.model.device)
                elif shape.index in self.meridians:
                    from .revolved_queries import inside_shape as inside_revolved
                    begin, profile_count, sectors = self.profile_layout_host[shape.index]
                    wp.launch(inside_revolved, dim=len(points), inputs=[gp, shape.pose,
                        wp.vec3(*shape.scale), self.profile_data, int(begin), int(profile_count), int(sectors)],
                        outputs=[inside], device=self.model.device)
                else:
                    wp.launch(inside_shape, dim=len(points), inputs=[gp, shape.pose, shape.kind,
                        wp.vec3(*shape.scale), wp.uint64(shape.mesh.id if shape.mesh else 0), shape.closed],
                        outputs=[inside], device=self.model.device)
                wp.launch(assign_owner, dim=len(points), inputs=[inside, shape.index],
                    outputs=[owners], device=self.model.device)
            owner = owners.numpy()
            if np.any(owner >= 0):
                yield points[owner >= 0], owner[owner >= 0]
