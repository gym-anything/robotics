"""Privileged geometric outcome checks. No action-name or image-based reward."""
import numpy as np
from scipy.spatial.transform import Rotation
from .containment import BagVolume
from .contact_evidence import ContactEvidence
from .variants import TASKS_V2_CONTRACT,CONTRACT_CHOICES

DESCRIPTION=('Deposit all designated empty beverage cans from the return counter and bussing tub '
    'into the recycling collection bag. Keep reusable items out of the recycling, prevent cans '
    'from falling onto the floor, leave the station upright and intact, and return the empty tub '
    'to the counter. Finish with the cans released and settled inside the bag, with nothing '
    'lodged across the inlet. Preserve all cans already in the bag.')
# deposit-cans/v1/high_fill/seed-0: the instruction states the job; every tolerance is returned in
# evaluation['thresholds']. `reference` keeps DESCRIPTION and v1 by
# default; its eight-task entry (contract deposit-tasks-v2, selected explicitly) uses this text.
HIGH_FILL_DESCRIPTION=('Deposit all the empty beverage cans from the return counter and the bussing tub '
    'into the recycling bag. Leave the reusable cups on their tray, and put the empty tub back on the counter.')
# single_can, sort_cups and empty_tub: the instruction states the end state only; how (one can at a time, batches,
# tipping the tub) is the policy's choice, and the high_fill contract's evaluator judges the end state.
TASK_DESCRIPTIONS={
    'single_can':('A guest left an empty beverage can on the return counter. Put it in the recycling bag. '
        'Leave the reusable cups on their tray and the cans already in the bag where they are.'),
    'sort_cups':('Three reusable cups were left among the empty beverage cans on the return counter. Put each cup back '
        'on the reusable tray and deposit all the empty cans in the recycling bag. Leave the cans already in the bag where they are.'),
    'empty_tub':('Empty the bussing tub: deposit all its empty beverage cans in the recycling bag, then put the empty tub '
        'back on the counter. Leave the reusable cups on their tray and the cans already in the bag where they are.')}
# lid_cans, sort_waste and tub_cups: the same rule (the end state only).
TASK_DESCRIPTIONS_V2={
    'lid_cans':('Guests left three empty cans standing on top of the recycling station\'s lid. Put them in the recycling bag. '
        'Leave the reusable cups on their tray and the cans already in the bag where they are.'),
    'sort_waste':('Guests left empty beverage cans and two used paper coffee cups on the return counter. Put the cans in the '
        'recycling bag and the paper cups in the landfill bin. Leave the reusable cups on their tray and the cans already '
        'in the bag where they are.'),
    'tub_cups':('The bussing tub came back with three reusable cups among the empty cans. Put the cups back on the reusable '
        'tray, deposit all the cans in the recycling bag, and put the empty tub back on the counter. Leave the cans already '
        'in the bag where they are.')}
# reference: v1 (the default) keeps DESCRIPTION; the eight-task entry (TASKS_V2_CONTRACT) states the job as high_fill does.
REFERENCE_DESCRIPTIONS={'v1':DESCRIPTION,TASKS_V2_CONTRACT:HIGH_FILL_DESCRIPTION}
DESCRIPTIONS={'reference':DESCRIPTION,'high_fill':HIGH_FILL_DESCRIPTION,**TASK_DESCRIPTIONS,**TASK_DESCRIPTIONS_V2}
# HIGH_FILL_CONTRACT: one held rule for tub, tray and cups (blocks success, no unit while in place), the
# tray judged against the counter top, and units for targets on the floor or in the landfill bin.
HIGH_FILL_CONTRACT='high-fill-v1'
TASKS_CONTRACT='deposit-tasks-v1'   # the high_fill evaluator (single_can, empty_tub)
# TASKS_V2_CONTRACT (variants.py), the eight-task catalogue: the high_fill evaluator plus, for the cups tasks
# (sort_cups, tub_cups), one credit per cup on the tray and one unit per cup in the bag or on the floor, and for
# sort_waste, the paper cups' landfill credit, their bag/floor units and the landfill bin upright. sort_cups
# uses it (its cups start on the right-hand counter row); single_can and empty_tub keep their own rule and
# contract; reference uses it only when selected explicitly (its eight-task entry), else v1.
TASKS_V2_VARIANTS=('lid_cans','sort_cups','sort_waste','tub_cups')
CUPS_TASKS=('sort_cups','tub_cups')   # the reusable cups start off the tray: put back is credited, not charged
# Default contract per variant (reference: TASKS_V2_CONTRACT on request, variants.CONTRACT_CHOICES).
TASK_CONTRACTS={'reference':'v1','high_fill':HIGH_FILL_CONTRACT,
    'single_can':TASKS_CONTRACT,'empty_tub':TASKS_CONTRACT,**{k:TASKS_V2_CONTRACT for k in TASKS_V2_VARIANTS}}
COUNTER_TOP_X=(-.70,-.06);COUNTER_TOP_Y=(-.69,.69)   # return_counter box top, assembly.py
TUB_REST_Z=.903                                        # tub origin is its base centre
TUB_HALF=(.1945,.253)                                  # tub_geometry outer footprint half-extents
# Bottom centres of the station's four rubber feet in the frame's local coordinates (station.py build).
STATION_FEET_LOCAL=np.array([(-.2965,-.2126,0.),(.2965,-.2126,0.),(-.3063,.2126,0.),(.3063,.2126,0.)])
STATION_FOOT_HEIGHT_M=.006
CUP_BASE_RADIUS=.028;CUP_RIM_RADIUS=.0375;CUP_HEIGHT=.132  # props.cup_geometry outer frustum
PAPER_CUP_BASE_RADIUS=.030;PAPER_CUP_RIM_RADIUS=.045;PAPER_CUP_HEIGHT=.110  # props PAPER_CUP_RINGS outer frustum (sort_waste)
TRAY_HALF=(.20,.09)                                    # props tray outer footprint
LANDFILL_HALF=(.254,.1397);LANDFILL_HEIGHT=.762        # props landfill_container body envelope, origin at its base
LANDFILL_UP_COS=.98                                    # sort_waste: the landfill bin left upright
TARGET_FLOOR_HEIGHT_M=.010                             # a target's lowest point below this is resting on the floor
REST_SECONDS=1.;REST_DISPLACEMENT_M=.010;REST_ROTATION_RAD=.05
PARTIAL_CREDIT=.9;UNFINISHED_SCORE=.9;SIGNED_PROGRESS_FLOOR=-1.

def description(variant,contract=None):
    """The instruction of a catalogue variant under its contract (None: its default; reference: v1)."""
    if variant not in DESCRIPTIONS:raise ValueError('Unknown deposit-cans variation: '+str(variant))
    if variant in CONTRACT_CHOICES:
        contract=contract or CONTRACT_CHOICES[variant][0]
        if contract not in CONTRACT_CHOICES[variant]:raise ValueError(f'Unknown task contract {contract!r} for {variant}')
        return REFERENCE_DESCRIPTIONS[contract]
    if contract is not None and contract!=TASK_CONTRACTS[variant]:raise ValueError(f'{variant} has only contract {TASK_CONTRACTS[variant]}')
    return DESCRIPTIONS[variant]

def task_contract(config):
    """The contract a configuration is judged by.

    config.task_contract, set by the catalogue (variants.resolve) only for a non-default choice, selects one of
    variants.CONTRACT_CHOICES; '' keeps the inventory's standing contract (a reference inventory: v1)."""
    from .variants import preset
    name=preset(config);chosen=getattr(config,'task_contract','')
    if chosen:
        if chosen not in CONTRACT_CHOICES.get(name,()) or (chosen!='v1' and config.scene_revision<7):
            raise ValueError(f'Task contract {chosen!r} is not selectable for this inventory and scene revision')
        return chosen
    if config.scene_revision<7:return 'v1'
    if name=='high_fill':return HIGH_FILL_CONTRACT
    if name in ('single_can','empty_tub'):return TASKS_CONTRACT
    if name in TASKS_V2_VARIANTS:return TASKS_V2_CONTRACT
    return 'v1'

def frustum_z_range(base,axis,base_radius,rim_radius,height):
    """Lowest and highest z of a cup's outer frustum from its base centre and unit axis, both in one frame."""
    lean=float(np.sqrt(max(0.,1.-float(axis[2])**2)));rim=float(base[2])+height*float(axis[2])
    return (min(float(base[2])-base_radius*lean,rim-rim_radius*lean),max(float(base[2])+base_radius*lean,rim+rim_radius*lean))

def inside_landfill(middle,highest):
    """An item in the landfill bin: its mid-axis point over the body envelope and above the base, its highest
    point below the rim (both in the bin frame, origin at the base centre)."""
    return bool(abs(middle[0])<LANDFILL_HALF[0] and abs(middle[1])<LANDFILL_HALF[1] and middle[2]>0. and highest<LANDFILL_HEIGHT)

def shell_area_inside_share(triangles,inside):
    """Share of a can's shell area whose face-centre probes are inside the closed bag volume.

    `triangles` is (faces, 3 vertices, 3) in current positions; `inside` the face-centre verdicts.
    It does not depend on whether the can enters end first, lying or tilted."""
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    return float(area[np.asarray(inside,bool)].sum()/max(float(area.sum()),1e-15))

def trailing_rest(history,tick,positions,quaternions,window_ticks):
    """Rest over the trailing hold window, measured from where things were when that window began.

    `history` is a list of dicts (tick, positions, quaternions), mutated in place: samples older
    than the latest one at or before `tick-window_ticks` are dropped. Returns whether the window is
    fully covered and every tracked position stayed within REST_DISPLACEMENT_M and every tracked
    rotation within REST_ROTATION_RAD of the window-start sample, plus diagnostics."""
    if history and history[-1]['tick']>=tick:del history[next(i for i,h in enumerate(history) if h['tick']>=tick):]
    history.append(dict(tick=int(tick),positions=np.asarray(positions,np.float64),quaternions=np.asarray(quaternions,np.float64)))
    start=tick-window_ticks;keep=0
    for i,h in enumerate(history):
        if h['tick']<=start:keep=i
    del history[:keep]
    reference=history[0]
    shifts=np.linalg.norm(np.stack([h['positions'] for h in history])-reference['positions'],axis=2).max(0)
    q=np.stack([h['quaternions'] for h in history]);k,m=q.shape[:2]
    turns=(Rotation.from_quat(q.reshape(-1,4))*Rotation.from_quat(np.tile(reference['quaternions'],(k,1))).inv()
        ).magnitude().reshape(k,m).max(0)
    covered=reference['tick']<=start
    resting=bool(covered and shifts.max()<=REST_DISPLACEMENT_M and turns.max()<=REST_ROTATION_RAD)
    return resting,dict(window_start_tick=reference['tick'],covers_full_window=bool(covered),
        displacement_m=float(shifts.max()),displacement_index=int(shifts.argmax()),rotation_rad=float(turns.max()))

class Progress:
    def __init__(self,env):
        self.contract=task_contract(env.config);self.rest_history=[]
        from .variants import preset
        self.cups_task=preset(env.config) in CUPS_TASKS
        # TASKS_V2_CONTRACT items whose mid-axis point is also classified against the deformed bag on every
        # update (one query with the can probes): the reusable cups of a cups task and the paper cups.
        # (body, base radius, rim radius, height); empty for every other contract.
        objects=env.assembly.objects;self.bag_items=[]
        if self.contract==TASKS_V2_CONTRACT:
            if self.cups_task and objects.get('preservation'):
                self.bag_items+=[(b,CUP_BASE_RADIUS,CUP_RIM_RADIUS,CUP_HEIGHT) for b in objects['preservation']['cups']]
            self.bag_items+=[(b,PAPER_CUP_BASE_RADIUS,PAPER_CUP_RIM_RADIUS,PAPER_CUP_HEIGHT) for b in objects.get('paper_cups',[])]
        self.bag=BagVolume(env);self.contacts=ContactEvidence(env)
        self.cans=[a for a in env.assembly.surfaces if a['kind']=='can']
        # Cache topology only. Every vertex/face-centre probe and every physical
        # substep is still checked against the current deformed bag.
        self.probe_spans=[];vertices=[];faces=[];vertex_rows=[];face_rows=[];offset=0
        for a in self.cans:
            n=a['count'];f=np.asarray(a['triangles'],np.int32)+a['first'];count=n+len(f)
            vertices.extend(range(a['first'],a['first']+n));faces.extend(f)
            vertex_rows.extend(range(offset,offset+n));face_rows.extend(range(offset+n,offset+count))
            self.probe_spans.append((offset,count));offset+=count
        self.probe_count=offset
        self.probe_vertices=np.asarray(vertices,np.int32);self.probe_faces=np.asarray(faces,np.int32).reshape(-1,3)
        self.probe_vertex_rows=np.asarray(vertex_rows,np.int32);self.probe_face_rows=np.asarray(face_rows,np.int32)
        self.can_faces=[np.asarray(a['triangles'],np.int32)+a['first'] for a in self.cans]
        self.stable_since=None;self.last_tick=-1;self.floor_losses=set()
        self.last_inside=None;self.last_points=None;self.last_mouth_z=None
        self.wall_transition_suspects=set();self.ejected_preloads=set()
        from .surface_integrity import SurfaceIntegrity
        faces=np.vstack([a['triangles']+a['first'] for a in env.assembly.surfaces])
        bag=env.assembly.bag_spec
        self.integrity=SurfaceIntegrity(env.state.particle_q.numpy(),faces,
            bag_ids=np.arange(bag['first'],bag['first']+bag['count']))
        self.surface_path_failures=([dict(tick=env.tick,reason='initial_surface_intersection')]
            if self.integrity.initial_intersection else [])
        self.surface_path_checked_through=env.tick
    def update(self,e):
        p=e.state.particle_q.numpy();pv=e.state.particle_qd.numpy()
        if e.tick>self.surface_path_checked_through:
            if e.tick!=self.surface_path_checked_through+1:
                self.surface_path_failures.append(dict(tick=e.tick,reason='missing_substep_path'))
            if not self.integrity.query(p):
                self.surface_path_failures.append(dict(tick=e.tick,reason='bag_surface_path_not_collision_free'))
            self.surface_path_checked_through=e.tick
        geometry_valid=not self.surface_path_failures
        q=e.state.body_q.numpy();v=e.state.body_qd.numpy()
        self.bag.update(p)
        frame=q[e.assembly.frame];R=Rotation.from_quat(frame[3:]);inverse=R.inv()
        a=e.assembly.bag_spec;mouth=p[a['first']+a['mouth']]
        mouth_local=inverse.apply(mouth-frame[:3]);mouth_z=float(mouth_local[:,2].min())
        points=np.empty((self.probe_count,3),dtype=p.dtype)
        points[self.probe_vertex_rows]=p[self.probe_vertices]
        if len(self.probe_faces):points[self.probe_face_rows]=p[self.probe_faces].mean(1)
        spans=self.probe_spans;local_points=inverse.apply(points-frame[:3])
        speeds=np.linalg.norm(pv,axis=1)
        query=points
        if self.bag_items:
            middles=[q[b,:3]+.5*h*Rotation.from_quat(q[b,3:]).apply([0,0,1]) for b,_,_,h in self.bag_items]
            query=np.vstack((points,np.asarray(middles,dtype=points.dtype)))
        inside,distance=self.bag.classify(query) if len(query) else (np.empty(0,bool),np.empty(0))
        item_in_bag={b:bool(x) for (b,*_),x in zip(self.bag_items,inside[self.probe_count:])}
        inside,distance=inside[:self.probe_count],distance[:self.probe_count]
        rows=[];wholly=[];stopped=True
        for a,(start,count) in zip(self.cans,spans):
            sl=slice(a['first'],a['first']+a['count']);local=local_points[start:start+count]
            below=bool(local[:,2].max()<mouth_z-.001)
            contained=bool(inside[start:start+count].all() and below)
            speed=float(speeds[sl].max());stopped &= speed<.03
            # Floor loss is retained even if the policy subsequently retrieves
            # the can. A can supported by the bag bottom is not floor loss.
            if not contained and float(p[sl,2].min())<.0015:self.floor_losses.add(a['name'])
            if a['origin']=='preload' and not contained:self.ejected_preloads.add(a['name'])
            rows.append(dict(id=a['name'],target=a['target'],inside=contained,
                below_mouth=below,max_speed_m_s=speed,maximum_signed_distance_m=float(distance[start:start+count].max())))
            wholly.append(contained)
        # This detects observed below-mouth side changes. The simulation also
        # applies its relative separating-plane/discovery bound to every
        # accepted nonlinear step. Neither is advertised as exact IPC CCD.
        if self.last_inside is not None and len(inside)==len(self.last_inside) and e.tick>self.last_tick:
            changed=inside!=self.last_inside
            z=local_points[:,2]
            oldz=inverse.apply(self.last_points-frame[:3])[:,2]
            suspect=changed&(np.maximum(z,oldz)<min(mouth_z,self.last_mouth_z)-.003)
            for a,(start,count) in zip(self.cans,spans):
                if suspect[start:start+count].any():self.wall_transition_suspects.add(a['name'])
        self.last_inside=inside.copy();self.last_points=points.copy();self.last_mouth_z=mouth_z;self.last_tick=e.tick
        active=self.contacts.read(robot_only=True) if e.tick else []
        held={r['surface'] for r in active};released=not any(a['name'] in held for a in self.cans)
        station_up=float(R.apply([0,0,1])[2])>.98
        station_ok=bool(station_up and -.025<frame[2]<.035 and np.linalg.norm(frame[:2]-[.49,0])<.10)
        pins=np.array([i[0] for i in e.assembly.pins]);local=np.array([i[2] for i in e.assembly.pins])
        pin_error=np.linalg.norm(p[pins]-(R.apply(local)+frame[:3]),axis=1)
        retained=bool(pin_error.max()<.005)
        if self.contract!='v1':
            return self._high_fill(e,p=p,q=q,v=v,frame=frame,rows=rows,wholly=wholly,points=points,inside=inside,
                released=released,station_up=station_up,retained=retained,pin_error=pin_error,
                geometry_valid=geometry_valid,vertices_still=stopped,item_in_bag=item_in_bag)
        tub=q[e.assembly.tub];up=Rotation.from_quat(tub[3:]).apply([0,0,1])
        tub_ok=bool(up[2]>.98 and np.linalg.norm(tub[:2]-[-.37,0])<.025 and abs(tub[2]-.903)<.012)
        reusable_ok=True
        preservation=e.assembly.objects.get('preservation')
        if preservation:
            tray=q[preservation['tray']]
            for body in preservation['cups']:
                pose=q[body];upright=Rotation.from_quat(pose[3:]).apply([0,0,1])[2]>.95
                relative=Rotation.from_quat(tray[3:]).inv().apply(pose[:3]-tray[:3])
                reusable_ok &= bool(upright and abs(relative[0])<.17 and abs(relative[1])<.055 and -.005<relative[2]<.035)
            reusable_ok &= bool(Rotation.from_quat(tray[3:]).apply([0,0,1])[2]>.98 and
                -.69<tray[0]<-.08 and abs(tray[1])<.60 and .88<tray[2]<.93)
        moving=[e.assembly.frame,e.assembly.tub]
        if preservation:moving += [preservation['tray'],*preservation['cups']]
        rigid_speed=np.linalg.norm(v[moving,:3],axis=1).max()
        rigid_angular_speed=np.linalg.norm(v[moving,3:],axis=1).max()
        stopped &= bool(rigid_speed<.03 and rigid_angular_speed<.15)
        target_rows=[r for r in rows if r['target']];total=len(target_rows);done=sum(r['inside'] for r in target_rows)
        eligible=bool(total and all(wholly) and released and station_ok and retained and tub_ok and reusable_ok and stopped
            and not self.floor_losses and not self.ejected_preloads and not self.wall_transition_suspects and not e.fault and geometry_valid)
        if not eligible:self.stable_since=None
        elif self.stable_since is None:self.stable_since=e.tick
        stable=0. if self.stable_since is None else (e.tick-self.stable_since)*e.config.dt
        return dict(success=eligible and stable>=1.,score=done/max(1,total),target_count=total,deposited_targets=done,
            cans=rows,station_upright=station_ok,retention_intact=retained,maximum_retention_error_m=float(pin_error.max()),
            tub_returned=tub_ok,reusables_preserved=bool(reusable_ok),released=released,stable_seconds=stable,
            settled=bool(stopped),maximum_preservation_linear_speed_m_s=float(rigid_speed),
            maximum_preservation_angular_speed_rad_s=float(rigid_angular_speed),
            floor_losses=sorted(self.floor_losses),ejected_preloads=sorted(self.ejected_preloads),
            observed_wall_transitions=sorted(self.wall_transition_suspects),bag_volume_m3=self.bag.volume_m3,
            physical_geometry_valid=geometry_valid,surface_path_failures=list(self.surface_path_failures),
            surface_path_checked_through=self.surface_path_checked_through,
            surface_path_method='IPC Toolkit1.6.0 TightInclusionCCD, default1e-6 tolerance; same linear substep adaptively certified on conservative rejection, maximum depth8',
            surface_path_query=dict(self.integrity.last_query),
            surface_path_scope='Bag self-intersection and can/bag crossings; no virtual cap, no can/can force changes',
            thresholds=dict(settle_seconds=1.,surface_speed_m_s=.03,retention_error_m=.005,
                preservation_linear_speed_m_s=.03,preservation_angular_speed_rad_s=.15),
            task_contract=self.contract,   # reported only; the v1 checks and score do not depend on it
            qualification='Component-qualified v6/scene7 with its exact outcome thresholds. No autonomous-solution or arbitrary-command guarantee; see the environment\'s report for limits.')

    def _high_fill(self,e,*,p,q,v,frame,rows,wholly,points,inside,released,station_up,retained,pin_error,
                   geometry_valid,vertices_still,item_in_bag=None):
        """HIGH_FILL_CONTRACT: the end state the instruction names, however and in whatever order it was reached.

        Containment, the 1 mm rim margin, release, bag retention, the wall-transition and
        swept-surface path checks are the v1 predicates. Floor contact and pre-existing cans
        that left the bag at some tick are reported; the end state decides (every can contained).
        """
        a=e.assembly;preservation=a.objects.get('preservation')
        # Station: upright and standing on its own four feet (each foot's base on the floor), so a
        # stand held or propped on two feet fails. Where on the floor it stands is not part of the job.
        feet=Rotation.from_quat(frame[3:]).apply(STATION_FEET_LOCAL)+frame[:3]
        station_ok=bool(station_up and np.abs(feet[:,2]).max()<STATION_FOOT_HEIGHT_M)
        # Tub: upright, resting at counter height, its centre over the counter top (anywhere on it),
        # and let go: no robot shape touching it.
        tub=q[a.tub];tub_turn=Rotation.from_quat(tub[3:]);tub_up=float(tub_turn.apply([0,0,1])[2])
        tub_placed=bool(tub_up>.98 and abs(tub[2]-TUB_REST_Z)<.012 and COUNTER_TOP_X[0]<tub[0]<COUNTER_TOP_X[1]
            and COUNTER_TOP_Y[0]<tub[1]<COUNTER_TOP_Y[1])
        # One "held" rule for the tub, tray and cups: an object a robot shape touches is not yet
        # put back or left (it blocks success) but is not damage (no unit while its pose is in place).
        paper=list(a.objects.get('paper_cups') or [])   # sort_waste's paper cups (TASKS_V2_CONTRACT), same rule
        held_bodies=[a.tub]+([preservation['tray'],*preservation['cups']] if preservation else [])+paper
        touched=self.contacts.robot_touching(held_bodies) if e.tick else set()
        tub_touched=a.tub in touched
        tub_ok=bool(tub_placed and not tub_touched)
        # Cups: each resting on its tray, standing, lying or nested in another cup that is; the tray
        # upright on the counter and not inside the tub (the tub is put back empty).
        cups_on_tray=[];tray_ok=True;tray_in_tub=False
        if preservation:
            tray=q[preservation['tray']];turn=Rotation.from_quat(tray[3:]);inverse=turn.inv();cups=[]
            for body in preservation['cups']:
                pose=q[body];axis=inverse.apply(Rotation.from_quat(pose[3:]).apply([0,0,1]))
                base=inverse.apply(pose[:3]-tray[:3]);middle=base+.5*CUP_HEIGHT*axis
                lean=float(np.sqrt(max(0.,1-axis[2]**2)))
                lowest=min(base[2]-CUP_BASE_RADIUS*lean,base[2]+CUP_HEIGHT*axis[2]-CUP_RIM_RADIUS*lean)
                over=bool(abs(middle[0])<TRAY_HALF[0] and abs(middle[1])<TRAY_HALF[1])
                cups.append(dict(base=base,axis=axis,over=over,ok=over and -.005<lowest<.035))
            for _ in cups:
                for c in cups:
                    if c['ok'] or not c['over']:continue
                    for d in cups:
                        if d is c or not d['ok']:continue
                        rel=c['base']-d['base'];along=float(rel@d['axis'])
                        radial=float(np.linalg.norm(rel-along*d['axis']))
                        if 0<along<CUP_HEIGHT and radial<CUP_BASE_RADIUS+(CUP_RIM_RADIUS-CUP_BASE_RADIUS)*along/CUP_HEIGHT \
                                and float(c['axis']@d['axis'])>.9:c['ok']=True
            cups_on_tray=[bool(c['ok']) for c in cups]
            in_tub=tub_turn.inv().apply(tray[:3]-tub[:3])
            tray_in_tub=bool(abs(in_tub[0])<TUB_HALF[0] and abs(in_tub[1])<TUB_HALF[1] and in_tub[2]>-.005)
            tray_ok=bool(turn.apply([0,0,1])[2]>.98 and COUNTER_TOP_X[0]<tray[0]<COUNTER_TOP_X[1]
                and COUNTER_TOP_Y[0]<tray[1]<COUNTER_TOP_Y[1] and .88<tray[2]<.93 and not tray_in_tub)
        reusables_held=[n for n,b in zip(['tray']+[f'cup_{k}' for k in range(len(cups_on_tray))],
            [preservation['tray'],*preservation['cups']] if preservation else []) if b in touched]
        reusable_ok=bool(all(cups_on_tray) and tray_ok and not reusables_held)
        # Reward: each target's share of its shell area inside the bag (full credit only when
        # contained), minus one unit per required item that is out of its place now.
        targets=[i for i,c in enumerate(self.cans) if c['target']];credit=[]
        for i in targets:
            row=rows[i];start,count=self.probe_spans[i];n=self.cans[i]['count']
            share=1. if row['inside'] else shell_area_inside_share(p[self.can_faces[i]],inside[start+n:start+count])
            value=0. if row['id'] in self.wall_transition_suspects else 1. if row['inside'] else PARTIAL_CREDIT*share
            row.update(inside_area_share=share,credit=value);credit.append(value)
        preloads_out=[r['id'] for c,r in zip(self.cans,rows) if c['origin']=='preload' and not r['inside']]
        # Targets spilled where the job cannot recover them or thrown in the wrong container, judged
        # by where they are now (no latch): in the landfill bin, or resting on the floor.
        in_bin=[];on_floor=[];bin_=a.objects.get('landfill_container')
        if bin_:bin_q=q[bin_['body']];bin_inv=Rotation.from_quat(bin_q[3:]).inv()
        for i in targets:
            if rows[i]['inside']:continue
            s=p[self.cans[i]['first']:self.cans[i]['first']+self.cans[i]['count']]
            rel=bin_inv.apply(s.mean(0)-bin_q[:3]) if bin_ else None
            if bin_ and abs(rel[0])<LANDFILL_HALF[0] and abs(rel[1])<LANDFILL_HALF[1] and rel[2]<LANDFILL_HEIGHT:in_bin.append(rows[i]['id'])
            elif float(s[:,2].min())<TARGET_FLOOR_HEIGHT_M:on_floor.append(rows[i]['id'])
        # TASKS_V2_CONTRACT items, judged by where they are now (no latch). A reusable cup of a cups task that is
        # not on the tray costs a unit inside the closed bag volume (its mid-axis point) or on the floor (lowest
        # point below TARGET_FLOOR_HEIGHT_M), where the job cannot recover it. Each of sort_waste's paper cups
        # earns one credit in the landfill bin and costs a unit in the bag or on the floor; the bin stays upright.
        eight=self.contract==TASKS_V2_CONTRACT;item_in_bag=item_in_bag or {}
        cups_lost=dict(in_bag=[],on_floor=[]);paper_rows=[];bin_up=None
        if eight and self.cups_task and preservation:
            for k,body in enumerate(preservation['cups']):
                if cups_on_tray[k]:continue
                pose=q[body];axis=Rotation.from_quat(pose[3:]).apply([0,0,1])
                if item_in_bag.get(body):cups_lost['in_bag'].append(f'cup_{k}')
                elif frustum_z_range(pose[:3],axis,CUP_BASE_RADIUS,CUP_RIM_RADIUS,CUP_HEIGHT)[0]<TARGET_FLOOR_HEIGHT_M:
                    cups_lost['on_floor'].append(f'cup_{k}')
        if paper:
            if not (eight and bin_):raise ValueError('Paper cups are judged only by '+TASKS_V2_CONTRACT+' with the landfill bin present')
            bin_up=float(Rotation.from_quat(bin_q[3:]).apply([0,0,1])[2])
            for k,body in enumerate(paper):
                pose=q[body];axis=Rotation.from_quat(pose[3:]).apply([0,0,1])
                base=bin_inv.apply(pose[:3]-bin_q[:3]);local_axis=bin_inv.apply(axis)
                in_landfill=inside_landfill(base+.5*PAPER_CUP_HEIGHT*local_axis,
                    frustum_z_range(base,local_axis,PAPER_CUP_BASE_RADIUS,PAPER_CUP_RIM_RADIUS,PAPER_CUP_HEIGHT)[1])
                bagged=bool(not in_landfill and item_in_bag.get(body))
                floored=bool(not in_landfill and not bagged and frustum_z_range(pose[:3],axis,PAPER_CUP_BASE_RADIUS,
                    PAPER_CUP_RIM_RADIUS,PAPER_CUP_HEIGHT)[0]<TARGET_FLOOR_HEIGHT_M)
                paper_rows.append(dict(id=f'paper_cup_{k}',in_landfill=in_landfill,in_bag=bagged,on_floor=floored,
                    touched_by_robot=body in touched,centre_m=[float(x) for x in pose[:3]+.5*PAPER_CUP_HEIGHT*axis]))
        paper_credit=float(sum(r['in_landfill'] for r in paper_rows))
        landfill_up=bool(bin_up is None or bin_up>LANDFILL_UP_COS)
        paper_ok=bool(all(r['in_landfill'] and not r['touched_by_robot'] for r in paper_rows) and landfill_up)
        collateral={k:n for k,n in (('pre_existing_cans_out_of_bag',len(preloads_out)),('targets_on_floor',len(on_floor)),
            ('targets_in_landfill',len(in_bin)),('tub_not_on_counter',int(not tub_placed)),
            ('cups_not_on_tray',sum(not x for x in cups_on_tray)),('tray_not_on_counter',int(not tray_ok)),
            ('station_not_upright',int(not station_ok)),('bag_not_retained',int(not retained)),
            ('cups_in_bag',len(cups_lost['in_bag'])),('cups_on_floor',len(cups_lost['on_floor'])),
            ('paper_cups_in_bag',sum(r['in_bag'] for r in paper_rows)),('paper_cups_on_floor',sum(r['on_floor'] for r in paper_rows)),
            ('landfill_not_upright',int(not landfill_up))) if n}
        cups_task=self.cups_task
        cup_credit=0.
        if cups_task:
            # The cups start off the tray (sort_cups: on the counter; tub_cups: in the tub): each one standing or
            # lying on the tray is part of the job (one credit), not damage; a cup elsewhere is simply not yet
            # done, except (TASKS_V2_CONTRACT) in the bag or on the floor, above.
            collateral.pop('cups_not_on_tray',None);cup_credit=float(sum(cups_on_tray))
        units=int(sum(collateral.values()));total=len(targets);done=sum(rows[i]['inside'] for i in targets)
        items=total+(len(cups_on_tray) if cups_task else 0)+len(paper_rows)
        net=float(sum(credit))+cup_credit+paper_credit-units;progress=float(min(1.,max(0.,net/max(1,items))))
        signed=float(min(1.,max(SIGNED_PROGRESS_FLOOR,net/max(1,items))))
        eligible=bool(total and all(wholly) and released and station_ok and retained and tub_ok and reusable_ok and paper_ok
            and not self.wall_transition_suspects and not e.fault and geometry_valid)
        # Rest: over the trailing hold window (the last REST_SECONDS), every target centre, the
        # station and the tub stay within REST_DISPLACEMENT_M / REST_ROTATION_RAD of where they were
        # when that window began. Slow creep, single-tick vertex or prop velocity pulses and the
        # untouched pre-existing cans' own packing therefore cannot veto a finished state.
        # sort_waste: each paper cup's centre (mid-axis point) is tracked too.
        bodies=[a.frame,a.tub]
        names=[self.cans[i]['name'] for i in targets]+['recycling_frame','bussing_tub']+[r['id'] for r in paper_rows]
        positions=np.vstack([p[self.cans[i]['first']:self.cans[i]['first']+self.cans[i]['count']].mean(0) for i in targets]+[q[bodies,:3]]
            +[np.asarray(r['centre_m'])[None] for r in paper_rows])
        resting=False;window={}
        if not eligible:self.rest_history=[];self.stable_since=None
        else:
            if self.stable_since is None:self.stable_since=e.tick
            resting,window=trailing_rest(self.rest_history,e.tick,positions,q[bodies,3:],int(round(REST_SECONDS/e.config.dt)))
            window['displacement_object']=names[window.pop('displacement_index')]
        stable=0. if self.stable_since is None else (e.tick-self.stable_since)*e.config.dt
        success=bool(eligible and resting)
        moving=bodies+([preservation['tray'],*preservation['cups']] if preservation else [])
        rigid_speed=float(np.linalg.norm(v[moving,:3],axis=1).max());rigid_angular=float(np.linalg.norm(v[moving,3:],axis=1).max())
        result=dict(success=success,score=1. if success else UNFINISHED_SCORE*signed,target_count=total,deposited_targets=done,
            cans=rows,station_upright=station_ok,station_feet_height_m=[float(z) for z in feet[:,2]],
            station_displacement_m=float(np.linalg.norm(frame[:2]-[.49,0])),
            retention_intact=retained,maximum_retention_error_m=float(pin_error.max()),
            tub_returned=tub_ok,tub_on_counter=tub_placed,tub_touched_by_robot=tub_touched,tub_position_m=[float(x) for x in tub[:3]],
            reusables_preserved=reusable_ok,cups_on_tray=cups_on_tray,tray_on_counter=tray_ok,tray_in_tub=tray_in_tub,
            reusables_held_by_robot=reusables_held,targets_on_floor=on_floor,targets_in_landfill=in_bin,released=released,
            stable_seconds=stable,settled=resting,rest_window=window,
            whole_pile_vertex_speed_settled_v1=bool(vertices_still and rigid_speed<.03 and rigid_angular<.15),
            maximum_preservation_linear_speed_m_s=rigid_speed,maximum_preservation_angular_speed_rad_s=rigid_angular,
            floor_losses=sorted(self.floor_losses),ejected_preloads=sorted(self.ejected_preloads),
            pre_existing_cans_not_contained=preloads_out,
            observed_wall_transitions=sorted(self.wall_transition_suspects),bag_volume_m3=self.bag.volume_m3,
            physical_geometry_valid=geometry_valid,surface_path_failures=list(self.surface_path_failures),
            surface_path_checked_through=self.surface_path_checked_through,
            surface_path_method='IPC Toolkit1.6.0 TightInclusionCCD, default1e-6 tolerance; same linear substep adaptively certified on conservative rejection, maximum depth8',
            surface_path_query=dict(self.integrity.last_query),
            surface_path_scope='Bag self-intersection and can/bag crossings; no virtual cap, no can/can force changes',
            reward=dict(contract=self.contract,progress=progress,signed_progress=signed,target_credit=float(sum(credit)),cup_credit=cup_credit,
                collateral_units=units,collateral=collateral,net_units=net,count_score_v1=done/max(1,total),
                score_rule=('1 if success else 0.9 x signed_progress; signed_progress = (target credit [+ cups put back, sort_cups] - collateral units)/(target_count [+ 3 cups, sort_cups]), '
                    'limited to [-1, 1] (below net -24 units the score stays at -0.9; use net_units); progress is the same clipped to [0, 1]')),
            task_contract=self.contract,
            thresholds=dict(hold_seconds=REST_SECONDS,rest_window_displacement_m=REST_DISPLACEMENT_M,
                rest_window_rotation_rad=REST_ROTATION_RAD,
                rest_window_rule='trailing: over the last hold_seconds, measured from each object\'s pose at the start of that window',
                rest_window_objects='each target can centre (mean shell vertex), station frame origin, tub origin',
                below_rim_margin_m=.001,retention_error_m=.005,station_up_cos=.98,station_feet_max_height_m=STATION_FOOT_HEIGHT_M,
                tub_up_cos=.98,tub_origin_z_m=[TUB_REST_Z-.012,TUB_REST_Z+.012],
                tub_centre_over_counter_top_m=dict(x=list(COUNTER_TOP_X),y=list(COUNTER_TOP_Y)),tub_touched_by_robot=False,
                cup_centre_within_tray_half_m=list(TRAY_HALF),cup_lowest_point_above_tray_base_m=[-.005,.035],
                cup_nested_in_cup_on_tray='base centre inside that cup frustum, axes within acos(0.9)',
                tray_up_cos=.98,tray_origin_over_counter_top_m=dict(x=list(COUNTER_TOP_X),y=list(COUNTER_TOP_Y)),tray_origin_z_m=[.88,.93],
                tray_origin_not_over_tub_footprint_half_m=list(TUB_HALF),
                held_rule='tub, tray or cup touched by a robot shape blocks success; no collateral unit while its pose is in place',
                target_on_floor_lowest_point_below_m=TARGET_FLOOR_HEIGHT_M,
                target_in_landfill=dict(centre_within_half_m=list(LANDFILL_HALF),below_m=LANDFILL_HEIGHT),
                wall_transition_band_m=.003,partial_credit='shell-area share of face-centre probes inside the bag',
                partial_credit_factor=PARTIAL_CREDIT,unfinished_score_factor=UNFINISHED_SCORE,
                signed_progress_floor=SIGNED_PROGRESS_FLOOR,reported_floor_contact_height_m=.0015),
            qualification=('task contract '+self.contract+' (the high_fill evaluator). Containment, release, '
                'retention and path checks are the v6/scene7 predicates. No autonomous-solution or arbitrary-command guarantee; see the environment\'s report.'))
        if eight:
            # Fields added only under TASKS_V2_CONTRACT; the HIGH_FILL_CONTRACT and TASKS_CONTRACT outputs do not carry them.
            result.update(cups_in_bag=cups_lost['in_bag'],cups_on_floor=cups_lost['on_floor'],paper_cups=paper_rows,
                landfill_upright=landfill_up,landfill_up_cos=bin_up,scored_items=items)
            result['reward'].update(paper_credit=paper_credit,score_rule=(
                '1 if success else 0.9 x signed_progress; signed_progress = (target credit + cups on the tray [sort_cups, tub_cups] '
                '+ paper cups in the landfill bin [sort_waste] - collateral units)/scored_items, scored_items = target_count '
                '[+ 3 cups, sort_cups and tub_cups] [+ 2 paper cups, sort_waste], limited to [-1, 1] (below net -scored_items the '
                'score stays at -0.9; use net_units); progress is the same clipped to [0, 1]'))
            result['thresholds'].update(
                rest_window_objects='each target can centre (mean shell vertex), station frame origin, tub origin, each paper cup centre (sort_waste)',
                held_rule='tub, tray, cup or paper cup touched by a robot shape blocks success; no collateral unit while its pose is in place',
                cup_in_bag='mid-axis point inside the closed bag volume (sort_cups, tub_cups: one unit per cup not on the tray)',
                cup_on_floor_lowest_point_below_m=TARGET_FLOOR_HEIGHT_M,
                paper_cup_in_landfill=dict(mid_axis_within_half_m=list(LANDFILL_HALF),mid_axis_above_base=True,highest_point_below_m=LANDFILL_HEIGHT),
                paper_cup_in_bag='mid-axis point inside the closed bag volume',paper_cup_on_floor_lowest_point_below_m=TARGET_FLOOR_HEIGHT_M,
                landfill_up_cos=LANDFILL_UP_COS)
        return result
