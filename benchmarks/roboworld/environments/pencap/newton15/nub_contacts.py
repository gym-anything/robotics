"""One contact spring per spherical retention nub, using exact mesh distance.

The native triangle manifold yields multiple constraints on one tiny sphere.
That duplicates the baseline's lumped spring. Only pen/nub pairs use this
adapter; other contacts use the native pipeline. Forces/friction/integration
remain Newton's solver. No axial guide, pose attraction or attachment is added.
"""

import numpy as np
import warp as wp
from newton._src.geometry.contact_data import ContactData
from newton._src.sim.collide import ContactWriterData, write_contact


@wp.kernel
def closest_nubs(pairs: wp.array[wp.vec2i], body_q: wp.array[wp.transform],
                 shape_body: wp.array[int], shape_transform: wp.array[wp.transform],
                 shape_source: wp.array[wp.uint64], shape_scale: wp.array[wp.vec3],
                 writer: ContactWriterData):
    tid = wp.tid()
    pen, nub = pairs[tid][0], pairs[tid][1]
    pen_body, nub_body = shape_body[pen], shape_body[nub]
    xp = shape_transform[pen]
    if pen_body >= 0:
        xp = body_q[pen_body]*xp
    xn = body_q[nub_body]*shape_transform[nub]
    center = wp.transform_get_translation(xn)
    local = wp.transform_point(wp.transform_inverse(xp), center)
    query = wp.mesh_query_point_sign_normal(shape_source[pen], local, .002)
    if query.result:
        surface = wp.mesh_eval_position(shape_source[pen], query.face, query.u, query.v)
        delta = local-surface
        distance = wp.length(delta)
        if distance > 1.e-9:
            normal = wp.transform_vector(xp, query.sign*delta/distance)
            radius = shape_scale[nub][0]
            a = wp.transform_point(xp, surface)
            b = center-radius*normal
            data = ContactData()
            data.contact_point_center = .5*(a+b)
            data.contact_normal_a_to_b = normal
            data.contact_distance = query.sign*distance-radius
            data.radius_eff_a = 0.
            data.radius_eff_b = 0.
            data.margin_a = 0.
            data.margin_b = 0.
            data.shape_a = pen
            data.shape_b = nub
            data.sort_sub_key = tid
            write_contact(data, writer, -1)


def filter_pairs(builder):
    pen = next((i for i, label in enumerate(builder.shape_label)
                if label in ("pen_shell", "barrel")), None)
    pairs = []
    if pen is not None:
        for i, label in enumerate(builder.shape_label):
            if "cap_nub" in label:
                builder.add_shape_collision_filter_pair(pen, i)
                pairs.append((pen, i))
    return pairs


class NubPipeline:
    def __init__(self, native, model, pairs):
        self.native, self.model = native, model
        self.pairs = wp.array(np.asarray(pairs, np.int32), dtype=wp.vec2i, device=model.device)
        self.empty_float = wp.empty(0, dtype=float, device=model.device)
        self.empty_key = wp.empty(0, dtype=wp.int64, device=model.device)

    def contacts(self):
        return self.native.contacts()

    def collide(self, state, contacts):
        self.native.collide(state, contacts)
        writer = ContactWriterData()
        writer.contact_max = contacts.rigid_contact_max
        writer.body_q = state.body_q
        writer.shape_body = self.model.shape_body
        writer.shape_gap = self.model.shape_gap
        writer.contact_count = contacts.rigid_contact_count
        for key, field in (("shape0", "shape0"), ("shape1", "shape1"), ("point0", "point0"),
            ("point1", "point1"), ("offset0", "offset0"), ("offset1", "offset1"),
            ("normal", "normal"), ("margin0", "margin0"), ("margin1", "margin1"), ("tids", "tids")):
            setattr(writer, "out_"+key, getattr(contacts, "rigid_contact_"+field))
        writer.out_stiffness = writer.out_damping = writer.out_friction = self.empty_float
        writer.out_sort_key = self.empty_key
        wp.launch(closest_nubs, dim=len(self.pairs), inputs=[self.pairs, state.body_q,
            self.model.shape_body, self.model.shape_transform, self.model.shape_source_ptr,
            self.model.shape_scale, writer], device=self.model.device)
        if int(contacts.rigid_contact_count.numpy()[0]) > contacts.rigid_contact_max:
            raise RuntimeError("Contact capacity exceeded; no contacts may be discarded")
