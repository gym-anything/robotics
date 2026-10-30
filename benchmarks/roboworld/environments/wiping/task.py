"""Privileged mass/geometry verifier. No rendered pixels or action names score."""
import numpy as np
from scipy.spatial.transform import Rotation

DESCRIPTION=('Reset this commercial preparation bench for the next food batch. Move the loose prep tray '
    'to wipe underneath it, remove crumbs into the collection pan or waste bin, and wipe the exposed '
    'stainless top, front edge and backsplash. Remove sticky food and oil without spreading them to the '
    'green clean-staging tray or floor. Use the damp cloths, dry cloth and scraper as needed; put used '
    'cloths in the blue bin. Leave the bench nearly dry, the tray upright on the bench, the tools on '
    'the trolley or in their collection pan, and release everything. The heavy blender is powered off; '
    'its permanent footprint is excluded. Cleaning is not a claim of microbiological sanitization.')

# Task revision 2, sticky-food preset only. The
# reference and oil-heavy presets keep DESCRIPTION and Progress unchanged.
REVISED_PRESETS=('sticky-food',)
STICKY_FOOD_DESCRIPTION=('Reset this commercial preparation bench for the next food batch; sanitizing is done '
    'separately afterwards. Clear the crumbs into the waste bin, and wipe the stainless top, front edge and '
    'backsplash, including the area under the loose prep tray. Remove the sticky food and oil without '
    'spreading them onto other surfaces, especially the green clean-staging tray or the floor, and leave the '
    'bench nearly dry. Use the damp cloths, dry cloth, scraper and collection pan as needed, and put used '
    'cloths in the blue bin. Leave the switched-off blender in place; the bench under it is already clean. '
    'Finish with the prep tray and the green tray on the bench, the scraper and collection pan back on the '
    'trolley, and release everything.')


def description(preset):
    """Instruction the policy receives for a soil preset."""
    return STICKY_FOOD_DESCRIPTION if preset in REVISED_PRESETS else DESCRIPTION


def progress(env):
    """Verifier for the environment's preset; component fixtures without soil keep Progress."""
    if env.transport is not None and env.config.soil_preset in REVISED_PRESETS:return BenchResetProgress(env)
    return Progress(env)


class Progress:
    def __init__(self,env):
        self.stable_since=None
        self.last_tick=-1
        self.initial_nonvolatile=env.transport.surface[:,1:].sum(0) if env.transport else np.ones(2)
        self.target_masks={}
        if env.transport:
            for i,s in enumerate(env.transport.surfaces):
                mask=np.ones(s.nx*s.ny,bool)
                if s.name=='worktop' and 'blender' in env.assembly.objects:
                    # The declared permanent footprint is excluded, not whatever
                    # area a policy later covers with a movable object.
                    obj=env.assembly.objects['blender'];x,y=np.meshgrid((np.arange(s.nx)+.5)*s.du,(np.arange(s.ny)+.5)*s.dv,indexing='ij')
                    world=s.origin+x.reshape(-1,1)*s.u+y.reshape(-1,1)*s.v
                    half=np.asarray(obj['footprint'])/2;center=np.asarray(obj['center'])[:2]
                    mask=~(abs(world[:,:2]-center)<half).all(1)
                self.target_masks[i]=mask

    def update(self,e):
        if e.transport is None:return dict(success=False,score=0.,reason='component fixture has no task')
        t=e.transport;q=e.state_0.body_q.numpy();v=e.state_0.body_qd.numpy()
        p=e.state_0.particle_q.numpy();pv=e.state_0.particle_qd.numpy()
        target=[];protected=[];other=[];peak=0.
        for i,s in enumerate(t.surfaces):
            values=t.surface[t.offsets[i]:t.offsets[i+1]]
            if s.name in ('worktop','front_edge','backsplash'):
                values=values[self.target_masks[i]]
                target.append(values.sum(0));peak=max(peak,float(values[:,1:].sum(1).max()/s.cell_area))
            elif s.protected:protected.append(values.sum(0))
            elif not s.name.startswith(('collection_pan','waste_bin','used_cloth_bin')):
                other.append(values.sum(0))
        remaining=np.sum(target,axis=0) if target else np.zeros(3)
        protected_mass=np.sum(protected,axis=0) if protected else np.zeros(3)
        misplaced=np.sum(other,axis=0) if other else np.zeros(3)
        if t.flow:misplaced+=t.flow.total()
        fractions=remaining[1:]/np.maximum(self.initial_nonvolatile,1e-30)
        def contained(points,name,radii=0.):
            obj=e.assembly.objects.get(name)
            if not obj:return np.zeros(len(points),bool)
            if obj['body']>=0:
                pose=q[obj['body']];local=Rotation.from_quat(pose[3:]).inv().apply(points-pose[:3])
            else:local=points-np.asarray(obj['center'])
            x,y,h=obj['size'];wall=obj['wall']
            return ((abs(local[:,0])+radii<x/2-wall)&(abs(local[:,1])+radii<y/2-wall)&
                    (local[:,2]-radii>=wall-.003)&(local[:,2]+radii<=h))
        used=[];disposed=[]
        for a in e.assembly.cloths:
            sl=slice(a['first'],a['first']+a['count'])
            if t.cloth[sl,1:].sum()>1e-7:
                used.append(a['name']);disposed.append(bool(contained(p[sl],'used_cloth_bin').all()))
        crumbs=e.assembly.objects.get('crumbs',[])
        cp=np.asarray([q[a['body'],:3] for a in crumbs]).reshape(-1,3)
        radii=np.asarray([a['radius'] for a in crumbs])
        captured=contained(cp,'collection_pan',radii)|contained(cp,'waste_bin',radii)
        crumbs_ok=bool(len(crumbs)>0 and captured.all())
        tray=e.assembly.objects.get('prep_tray');tray_ok=False
        if tray:
            pose=q[tray['body']];up=Rotation.from_quat(pose[3:]).apply([0,0,1])
            x,y,h=tray['size']
            tray_ok=bool(up[2]>.98 and abs(pose[0])<e.config.table_depth/2-x/2 and
                abs(pose[1])<e.config.table_width/2-y/2 and abs(pose[2]-e.config.table_height)<.015)
        fingers=[s['q'] for n,s in e.joints.items() if 'finger' in n]
        released=bool(np.all(e.state_0.joint_q.numpy()[fingers]>.03))
        scraper=e.assembly.objects.get('scraper');tool_ok=False
        if scraper:
            point=q[scraper['body'],:3]
            tool_ok=bool((-.89<point[0]<-.40 and -.62<point[1]<.74 and .82<point[2]<.95) or
                         contained(point[None],'collection_pan')[0])
        equipment_ok=True
        for name in ('blender','clean_staging_tray'):
            obj=e.assembly.objects.get(name)
            if obj is None:equipment_ok=False;continue
            pose=q[obj['body']];up=Rotation.from_quat(pose[3:]).apply([0,0,1])
            if name=='blender':
                equipment_ok &= bool(up[2]>.98 and np.linalg.norm(pose[:2]-np.asarray(obj['center'][:2]))<.02
                                     and abs(pose[2]-e.config.table_height)<.015)
            else:
                x,y,_=obj['size']
                equipment_ok &= bool(up[2]>.98 and abs(pose[0])<e.config.table_depth/2-x/2
                    and abs(pose[1])<e.config.table_width/2-y/2 and abs(pose[2]-e.config.table_height)<.015)
        clean=bool(np.all(fractions<=.01) and peak<=.0005 and remaining[0]<=.0005)
        containment_ok=bool(protected_mass[1:].sum()+misplaced[1:].sum()<=.001*self.initial_nonvolatile.sum())
        stopped=bool(np.linalg.norm(pv,axis=1).max()<.05 and np.linalg.norm(v[:,:3],axis=1).max()<.03
                     and np.linalg.norm(v[:,3:],axis=1).max()<.15)
        eligible=clean and containment_ok and crumbs_ok and all(disposed) and tray_ok and released and tool_ok and stopped and equipment_ok
        if not eligible:self.stable_since=None
        elif self.stable_since is None:self.stable_since=e.tick
        stable_s=0. if self.stable_since is None else (e.tick-self.stable_since)*e.config.dt
        return dict(success=bool(eligible and stable_s>=.5 and not e.fault),
            score=float(np.clip(1-fractions.mean(),0,1)*.7+float(crumbs_ok)*.1+float(all(disposed) and bool(used))*.1+
                        float(clean and containment_ok and released and tool_ok and tray_ok)*.1),
            remaining_target_kg=remaining.tolist(),nonvolatile_remaining_fractions=fractions.tolist(),
            peak_nonvolatile_kg_m2=peak,protected_surface_kg=protected_mass.tolist(),
            misplaced_kg=misplaced.tolist(),used_cloths=used,used_cloths_disposed=all(disposed),
            crumbs_collected=int(captured.sum()),crumbs_total=len(crumbs),tray_returned=tray_ok,
            grippers_released=released,tools_stowed=tool_ok,stable_seconds=stable_s,
            protected_equipment_upright=equipment_ok,
            thresholds=dict(nonvolatile_fraction=.01,peak_kg_m2=.0005,water_kg=.0005,
                            cross_contamination_fraction=.001,settling_s=.5),
            limitation='Visible soil surrogate, not allergens, microbes or a disinfection assay')


TARGET_FACES=('worktop','front_edge','backsplash')
# Only the two bins receive waste. The collection pan is a tool that goes back on the clean trolley, so
# soil on any of its faces is spread soil, like soil on the scraper.
CAPTURE_RECEPTACLES=('waste_bin','used_cloth_bin')
SUPPLY_TRAYS=('clean_cloth_tray_0','clean_cloth_tray_1','clean_cloth_tray_2')
# Verifier documentation for task revision 2. None of these numbers belongs in the instruction.
TOLERANCES=dict(
    nonvolatile_fraction=.01,           # each of food solids and oil left on the target faces, of its tick-0 mass
    peak_kg_m2=.0001,                   # food+oil in every target cell. optics.py tints a cell 1-exp(-food/1.5 g/m2)
                                        # toward the residue colour: 6.5% at 0.1 g/m2, 28% at the former 0.5 g/m2
    water_kg=.0005,                     # free water summed over all target faces ("nearly dry")
    cross_contamination_fraction=.001,  # food+oil on any surface other than the target, cloths and the two bins'
                                        # interiors, plus runoff in flight, unmapped or escaped, of tick-0 food+oil
    used_cloth_food_oil_kg=1e-7,        # a cloth holding more food+oil than this is used wherever it lies
    used_cloth_water_change_kg=2e-4,    # a cloth whose water differs this much from its tick-0 water has been used;
                                        # untouched cloths in their trays drift 6 mg in 60 s (rest240_457)
    supply_tray_heap_m=.10,             # an unused cloth may heap this high above its supply-tray floor
    receptacle_contact_m=.001,          # a crumb may touch or press up to this far into a bin wall plane
    receptacle_floor_allowance_m=.003,  # contents may sit this far below a receptacle's floor top (contact compliance)
    upright_cos=.98,                    # prep tray, green tray and blender: body z axis . world z
    resting_height_m=.015,              # trays and blender rest on the bench top
    blender_displacement_m=.02,
    blender_full_damage_m=.12,          # partial credit only: blender damage grows from 0 at 20 mm to full at 120 mm
    trolley_top_outline_m=((-.90,-.40),(-.64,.74)),  # scraper and collection-pan centre of mass over the trolley top
    trolley_resting_origin_z_m=(.82,.90),            # their body origins (blade / floor underside) resting on the top
                                                     # or on something on it (in the pan 0.842, on a clean cloth about
                                                     # 0.853, across a tray rim 0.862, pan inverted 0.875 m)
    trolley_cloth_z_m=(.835,.95),                    # an unused cloth may also rest directly on the trolley top
    grip_squeeze_m=.0001,               # a finger held this much wider than its servo set-point is squeezing an object:
                                        # 0.35 N at the 3500 N/m finger servo; two pads at mu 0.45 then hold < 0.32 N,
                                        # less than the lightest free object (0.59 N scraper, 0.60 N dry cloth)
    particle_speed_m_s=.05,body_speed_m_s=.03,body_rate_rad_s=.15,settling_s=.5,evaluation_interval_s=.1)
RULES=(
    'crumbs: every crumb centre, inset by its radius, inside the waste-bin inner box; it may press receptacle_contact_m '
    'into a wall plane and sit receptacle_floor_allowance_m below the floor top',
    'cloth in the blue bin: every vertex inside the bin inner box',
    'unused cloth: <= used_cloth_food_oil_kg food+oil, water within used_cloth_water_change_kg of tick 0, and every vertex '
    'either inside one supply-tray inner box up to supply_tray_heap_m above its floor, or over the trolley-top outline '
    'within trolley_cloth_z_m; every other cloth (including one on the floor) is used',
    'capture: cloths and the interior faces of the waste and blue bins; a face is interior if its centre, 1 mm along its '
    'normal, lies in the open cavity; the collection pan is a tool, not a capture receptacle',
    'trays on the bench: upright, body origin (base centre) over the worktop outline at any yaw, base within '
    'resting_height_m of the bench top',
    'blender in place: upright, base within resting_height_m of the bench top, within blender_displacement_m of its start',
    'scraper and collection pan on the trolley: centre of mass over trolley_top_outline_m and body origin within '
    'trolley_resting_origin_z_m, i.e. resting on the top surface (not the lower shelf) or on anything on it: in the pan, on '
    'a supply tray or on an unused cloth; any orientation. Soil a tool carries counts as spread; soil it passes to a cloth '
    'makes that cloth used',
    'released: no finger joint more than grip_squeeze_m wider than its current servo set-point, at any opening',
    'stopped: every cloth particle below particle_speed_m_s and every body below body_speed_m_s and body_rate_rad_s, '
    'except crumbs that pass the waste-bin test (their containment must itself hold for settling_s)',
    'eligibility is sampled every evaluation_interval_s and at each action end; success needs settling_s of continuous '
    'eligibility and no fault',
    'score = 1.0 exactly when success, else 0.9 x partial; partial = clip(sum(WEIGHTS x terms), 0, 1) x '
    '(1 - sum(PENALTIES x damage severity)) x (1 - min(1, spread food+oil / tick-0 food+oil)); severities are 0..1 and '
    'present only while a start-state condition is broken; soil_captured is food+oil in cloths and bin interiors / tick 0; '
    'dryness counts target water removed minus water newly spread off the target (success itself checks target water only)')
WEIGHTS=dict(soil_captured=.45,visibly_clean_area=.15,dryness=.10,crumbs=.15,used_cloths_binned=.15)
PENALTIES=dict(blender_moved=.10,staging_tray_off_bench=.10,prep_tray_off_bench=.05,
               scraper_off_trolley=.025,collection_pan_off_trolley=.025)


class BenchResetProgress:
    """Task revision 2 verifier and reward (sticky-food preset). Deterministic state functions only.

    Success (held 0.5 s, no fault) needs all of:
      clean      food and oil each <= 1% on worktop, front edge and backsplash; every cell <= 0.1 g/m2
                 food+oil; <= 0.5 g water. The blender footprint is ordinary target bench: no seed
                 authors soil there, so only soil a policy pushes under the blender is found there.
      contained  food+oil off the target, outside cloths and the two bins' INTERIORS, plus runoff, <= 0.1%.
                 The collection pan is a tool: soil on it is spread soil.
      crumbs     every crumb inside the waste bin (touching a wall counts as inside).
      cloths     every cloth is wholly in the blue bin, or is unused: no food/oil, unchanged water, in a supply
                 tray or on the trolley top.
      equipment  prep and green trays upright, centre over the bench, resting on it (any yaw); blender
                 upright, within 20 mm, at bench height; scraper and collection pan resting on the trolley top
                 or on anything on it, in any orientation.
      released   no finger is held open wider than its servo set-point (nothing squeezed), any opening.
      stopped    particles < 0.05 m/s, bodies < 0.03 m/s and < 0.15 rad/s, except crumbs already in the waste bin.
    score is 1.0 exactly when success is true; otherwise 0.9 x partial, a continuous end-state
    account of the instruction's clauses (WEIGHTS), scaled down by graded damage while a start-state
    condition is broken (PENALTIES) and by the fraction of food+oil spread off the target. Nothing
    depends on actions, order, tools used or intermediate milestones. Remaining rules are listed in
    RULES and returned with the thresholds.
    """
    revision=2

    def __init__(self,env):
        t=env.transport;self.stable_since=None
        groups=dict(target=[],area=[],capture=[],protected=[],other=[])
        for i,s in enumerate(t.surfaces):
            rows=np.arange(t.offsets[i],t.offsets[i+1])
            if s.name in TARGET_FACES:groups['target'].append(rows);groups['area'].append(np.full(len(rows),s.cell_area))
            elif s.name.startswith(CAPTURE_RECEPTACLES) and self._interior(env,s):groups['capture'].append(rows)
            elif s.protected:groups['protected'].append(rows)
            else:groups['other'].append(rows)
        # Derived geometry, not episode state (excluded from checkpoints like Progress.target_masks).
        self.geometry={k:np.concatenate(v) if v else np.zeros(0,int) for k,v in groups.items()}
        g=self.geometry;g['offtarget']=np.concatenate((g['protected'],g['other']))
        g['offtarget_area']=np.asarray([t.surfaces[i].cell_area for i in np.searchsorted(t.offsets,g['offtarget'],side='right')-1])
        g['com']=np.asarray(env.assembly.builder.body_com,float).reshape(-1,3)
        obj=env.assembly.objects['blender'];w=self._cell_centres(t,g['target'])
        g['footprint']=(abs(w[:,:2]-np.asarray(obj['center'])[:2])<np.asarray(obj['footprint'])/2).all(1)
        initial=t.surface[g['target']]
        self.initial_nonvolatile=initial[:,1:].sum(0)
        self.initial_water=float(initial[:,0].sum())
        self.initial_dirty_area=self._dirty_area(initial[:,1:].sum(1))
        self.initial_offtarget_dirty_area=self._dirty_area(t.surface[g['offtarget'],1:].sum(1),g['offtarget_area'])
        self.initial_offtarget_water=float(t.surface[g['offtarget'],0].sum()+(t.flow.total()[0] if t.flow else 0.))
        self.initial_cloth_water={a['name']:float(t.cloth[a['first']:a['first']+a['count'],0].sum()) for a in env.assembly.cloths}

    @staticmethod
    def _interior(env,s):
        """Receptacle face whose centre, 1 mm along its normal, lies in the open cavity."""
        name=next(n for n in CAPTURE_RECEPTACLES if s.name.startswith(n))
        obj=env.assembly.objects[name];x,y,h=obj['size'];wall=obj['wall']
        p=s.origin+s.u*s.size[0]/2+s.v*s.size[1]/2+.001*s.normal
        if obj['body']<0:p=p-np.asarray(obj['center'])
        return bool(abs(p[0])<=x/2-wall+1e-6 and abs(p[1])<=y/2-wall+1e-6 and wall-1e-6<=p[2]<=h)

    @staticmethod
    def _cell_centres(t,rows):
        out=np.zeros((len(rows),3));face=np.searchsorted(t.offsets,rows,side='right')-1
        for i in np.unique(face):
            s=t.surfaces[i];k=rows[face==i]-t.offsets[i]
            out[face==i]=s.origin+((k//s.ny+.5)*s.du)[:,None]*s.u+((k%s.ny+.5)*s.dv)[:,None]*s.v
        return out

    def _dirty_area(self,nonvolatile,area=None):
        """Area equivalent at or above the residue ceiling; lighter cells count proportionally."""
        area=self.geometry['area'] if area is None else area
        return float(np.minimum(area,nonvolatile/TOLERANCES['peak_kg_m2']).sum())

    @staticmethod
    def _inside(e,q,name,points,radii=0.,top=None,contact=0.):
        obj=e.assembly.objects.get(name)
        if not obj:return np.zeros(len(points),bool)
        if obj['body']>=0:pose=q[obj['body']];local=Rotation.from_quat(pose[3:]).inv().apply(points-pose[:3])
        else:local=points-np.asarray(obj['center'])
        x,y,h=obj['size'];wall=obj['wall'];top=h if top is None else top
        return ((abs(local[:,0])+radii<x/2-wall+contact)&(abs(local[:,1])+radii<y/2-wall+contact)&
                (local[:,2]-radii>=wall-TOLERANCES['receptacle_floor_allowance_m'])&(local[:,2]+radii<=top))

    @staticmethod
    def _servo_targets(e):
        control=getattr(e,'control',None);target=None if control is None else control.joint_target_q
        return np.asarray(e.targets if target is None else target.numpy())

    def update(self,e):
        t=e.transport;g=self.geometry;T=TOLERANCES;c=e.config;objects=e.assembly.objects
        q=e.state_0.body_q.numpy();v=e.state_0.body_qd.numpy()
        p=e.state_0.particle_q.numpy();pv=e.state_0.particle_qd.numpy()
        target=t.surface[g['target']];remaining=target.sum(0);nonvolatile=target[:,1:].sum(1)
        peak=float((nonvolatile/g['area']).max())
        receptacle=t.surface[g['capture']].sum(0);protected_mass=t.surface[g['protected']].sum(0)
        misplaced=t.surface[g['other']].sum(0)+(t.flow.total() if t.flow else 0.)
        fractions=remaining[1:]/np.maximum(self.initial_nonvolatile,1e-30)
        up=lambda body:float(Rotation.from_quat(q[body,3:]).apply([0,0,1])[2])
        com=lambda body:q[body,:3]+Rotation.from_quat(q[body,3:]).apply(g['com'][body])
        (tx0,tx1),(ty0,ty1)=T['trolley_top_outline_m'];tz0,tz1=T['trolley_resting_origin_z_m']
        def on_trolley(body):
            centre=com(body)
            return bool(tx0<centre[0]<tx1 and ty0<centre[1]<ty1 and tz0<q[body,2]<tz1)
        def on_bench(name):
            obj=objects.get(name)
            if obj is None:return False
            pose=q[obj['body']]
            return bool(up(obj['body'])>T['upright_cos'] and abs(pose[0])<c.table_depth/2 and
                        abs(pose[1])<c.table_width/2 and abs(pose[2]-c.table_height)<T['resting_height_m'])
        tray_severity=lambda name,ok:0. if ok else (.5 if up(objects[name]['body'])>T['upright_cos'] else 1.)
        cloths={}
        for a in e.assembly.cloths:
            sl=slice(a['first'],a['first']+a['count']);soil=float(t.cloth[sl,1:].sum())
            water_change=float(t.cloth[sl,0].sum())-self.initial_cloth_water.get(a['name'],0.)
            in_bin=self._inside(e,q,'used_cloth_bin',p[sl])
            in_tray=any(self._inside(e,q,n,p[sl],top=objects[n]['wall']+T['supply_tray_heap_m']).all()
                        for n in SUPPLY_TRAYS if n in objects)
            on_top=bool(((p[sl,0]>tx0)&(p[sl,0]<tx1)&(p[sl,1]>ty0)&(p[sl,1]<ty1)&
                         (p[sl,2]>T['trolley_cloth_z_m'][0])&(p[sl,2]<T['trolley_cloth_z_m'][1])).all())
            used=soil>T['used_cloth_food_oil_kg'] or abs(water_change)>T['used_cloth_water_change_kg'] or not (in_tray or on_top)
            cloths[a['name']]=dict(used=bool(used),in_blue_bin=bool(in_bin.all()),in_supply_tray=bool(in_tray),
                                   on_trolley_top=on_top,food_oil_kg=soil,
                                   water_change_kg=water_change,vertex_fraction_in_blue_bin=float(in_bin.mean()),
                                   food_oil_in_blue_bin_kg=float(t.cloth[sl][in_bin,1:].sum()))
        cloths_ok=all(x['in_blue_bin'] or not x['used'] for x in cloths.values())
        crumbs=objects.get('crumbs',[])
        cp=np.asarray([q[a['body'],:3] for a in crumbs]).reshape(-1,3);radii=np.asarray([a['radius'] for a in crumbs])
        captured=self._inside(e,q,'waste_bin',cp,radii,contact=T['receptacle_contact_m'])
        crumbs_ok=bool(len(crumbs)>0 and captured.all())
        tray_ok=on_bench('prep_tray');staging_ok=on_bench('clean_staging_tray')
        blender=objects['blender'];bp=q[blender['body']]
        blender_offset=float(np.linalg.norm(bp[:2]-np.asarray(blender['center'][:2])))
        blender_standing=bool(up(blender['body'])>T['upright_cos'] and abs(bp[2]-c.table_height)<T['resting_height_m'])
        blender_ok=bool(blender_standing and blender_offset<T['blender_displacement_m'])
        # Since crumbs count only in the waste bin, the pan's orientation guards nothing.
        scraper_ok=on_trolley(objects['scraper']['body']);pan_ok=on_trolley(objects['collection_pan']['body'])
        fingers=[s['q'] for n,s in e.joints.items() if 'finger' in n]
        squeeze=e.state_0.joint_q.numpy()[fingers]-self._servo_targets(e)[fingers]
        released=bool(np.all(squeeze<=T['grip_squeeze_m']))
        clean=bool(np.all(fractions<=T['nonvolatile_fraction']) and peak<=T['peak_kg_m2'] and remaining[0]<=T['water_kg'])
        spread=float(protected_mass[1:].sum()+misplaced[1:].sum())
        containment_ok=bool(spread<=T['cross_contamination_fraction']*self.initial_nonvolatile.sum())
        # Crumbs already in the fixed waste bin cannot change any predicate by moving; their containment itself
        # must hold for settling_s. Their residual contact spin is an environment defect.
        still=np.ones(len(v),bool);still[[a['body'] for a,inside in zip(crumbs,captured) if inside]]=False
        stopped=bool(np.linalg.norm(pv,axis=1).max()<T['particle_speed_m_s'] and np.linalg.norm(v[still,:3],axis=1).max()<T['body_speed_m_s']
                     and np.linalg.norm(v[still,3:],axis=1).max()<T['body_rate_rad_s'])
        predicates=dict(clean=clean,contained=containment_ok,crumbs_collected=crumbs_ok,cloths_disposed=cloths_ok,
                        prep_tray_on_bench=tray_ok,staging_tray_on_bench=staging_ok,blender_in_place=blender_ok,
                        scraper_on_trolley=scraper_ok,collection_pan_on_trolley=pan_ok,released=released,stopped=stopped)
        eligible=all(predicates.values())
        if not eligible:self.stable_since=None
        elif self.stable_since is None:self.stable_since=e.tick
        stable_s=0. if self.stable_since is None else (e.tick-self.stable_since)*c.dt
        success=bool(eligible and stable_s>=T['settling_s'] and not e.fault)
        m0=float(self.initial_nonvolatile.sum());captured_soil=float(t.cloth[:,1:].sum()+receptacle[1:].sum())
        soil=captured_soil/m0
        # Spread soil scales credit like damage instead of being subtracted before the clip, so later
        # progress stays visible after a spill.
        spread_factor=1-min(1.,spread/m0)
        # Newly soiled area anywhere off the target counts like dirty bench area.
        offtarget_area=max(0.,self._dirty_area(t.surface[g['offtarget'],1:].sum(1),g['offtarget_area'])-self.initial_offtarget_dirty_area)
        clean_area=float(np.clip(1-(self._dirty_area(nonvolatile)+offtarget_area)/max(self.initial_dirty_area,1e-30),0,1))
        used=[n for n,x in cloths.items() if x['used']]
        binned_cloths=float(np.mean([cloths[n]['vertex_fraction_in_blue_bin'] for n in used])) if used else 1.
        # Drying credit counts water taken off the target into cloths or bins, not water pushed elsewhere, so
        # moving a smear off the bench is never net progress under the spread factor.
        water_spread=max(0.,float(protected_mass[0]+misplaced[0])-self.initial_offtarget_water)
        terms=dict(soil_captured=soil,visibly_clean_area=min(clean_area,soil),
                   dryness=float(np.clip((self.initial_water-remaining[0]-water_spread)/max(self.initial_water-T['water_kg'],1e-30),0,1)),
                   crumbs=float(captured.mean()) if len(crumbs) else 0.,used_cloths_binned=min(binned_cloths,soil))
        severity=dict(blender_moved=0. if blender_ok else 1. if not blender_standing else
                          float(np.clip((blender_offset-T['blender_displacement_m'])/(T['blender_full_damage_m']-T['blender_displacement_m']),0,1)),
                      staging_tray_off_bench=tray_severity('clean_staging_tray',staging_ok),prep_tray_off_bench=tray_severity('prep_tray',tray_ok),
                      scraper_off_trolley=float(not scraper_ok),collection_pan_off_trolley=float(not pan_ok))
        damage=sum(PENALTIES[k]*x for k,x in severity.items())
        partial=float(np.clip(sum(WEIGHTS[k]*x for k,x in terms.items()),0,1)*(1-damage)*spread_factor)
        disposed=(sum(x['food_oil_in_blue_bin_kg'] for x in cloths.values())+float(receptacle[1:].sum()))/m0
        return dict(success=success,score=1. if success else .9*partial,task_revision=self.revision,
            partial_score=partial,score_terms=terms,penalties=[k for k,x in severity.items() if x>0],
            damage_severity=severity,damage_factor=1-damage,spread_factor=spread_factor,predicates=predicates,
            infrastructure_valid=e.fault is None,
            remaining_target_kg=remaining.tolist(),nonvolatile_remaining_fractions=fractions.tolist(),
            peak_nonvolatile_kg_m2=peak,blender_footprint_food_oil_kg=float(nonvolatile[g['footprint']].sum()),
            protected_surface_kg=protected_mass.tolist(),misplaced_kg=misplaced.tolist(),
            receptacle_interior_kg=receptacle.tolist(),cloth_kg=t.cloth.sum(0).tolist(),
            ledger_fractions=dict(on_target=float(nonvolatile.sum())/m0,captured=captured_soil/m0,spread=spread/m0,disposed=disposed),
            offtarget_dirty_area_m2=offtarget_area,
            cloths=cloths,used_cloths=used,used_cloths_disposed=cloths_ok,
            crumbs_collected=int(captured.sum()),crumbs_total=len(crumbs),tray_returned=tray_ok,
            grippers_released=released,tools_stowed=bool(scraper_ok and pan_ok),stable_seconds=stable_s,
            protected_equipment_upright=bool(blender_ok and staging_ok),
            thresholds=dict(TOLERANCES,weights=WEIGHTS,penalties=PENALTIES,failure_score_scale=.9,rules=RULES),
            limitation='Visible soil surrogate, not allergens, microbes or a disinfection assay')
