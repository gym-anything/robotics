"""Architectural finish on existing collision envelopes, not task geometry.

Tile joints, cabinet fronts, fasteners and appliance trim are render-only relief
on the corresponding rigid collision envelope. No visual-only cleanup tools,
hidden grasp helpers or nonphysical dirt-removal animations are introduced.
"""
import os
import numpy as np
from pxr import UsdGeom,UsdShade,Gf,Tf
from whiteboard.usd_scene import bevel_box
from pencap.newton15.render import pose


def finish(stage,sim,material,mats):
    if sim.config.scene!='kitchen':return
    material('grout',(.24,.27,.26),.9,0.)
    material('tile_cream',(.64,.66,.62),.3,0.)
    material('floor_grout',(.15,.18,.17),.95,0.)
    material('brushed_trim',(.3,.34,.35),.39,.85)
    material('rubber',(.016,.022,.024),.85,0.)
    material('off_switch',(.13,.025,.018),.5,0.)
    def box(name,position,size,mat,parent='/Lab/finish',bevel=.0004):
        g=bevel_box(stage,parent+'/'+Tf.MakeValidIdentifier(name),size,bevel);pose(g,position)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[mat]);return g
    # Metric-scale scanned tiles bind to the actual wall/floor envelopes;
    # duplicate procedural tiles/grout would cover or contradict that finish.
    box('wall_skirt_back',(1.369,0,.065),(.014,4.8,.13),'brushed_trim')
    box('wall_skirt_side',(0,2.319,.065),(3.,.014,.13),'brushed_trim')
    # Flush service-counter fronts and handles: cabinet volume already collides.
    for i in range(4):
        y=-1.23+i*.45
        box(f'cabinet_front_{i}',(.886,y,.48),(.007,.44,.73),'brushed_trim',bevel=.002)
        box(f'cabinet_handle_{i}',(.871,y,.76),(.025,.14,.013),'charcoal',bevel=.003)
    for y in (-.78,.78):
        for x in (-.115,.115):
            for offset in (-.115,.115):
                sphere=UsdGeom.Sphere.Define(stage,f'/Lab/finish/bolt_{str(y).replace("-","m").replace(".","p")}_{str(x).replace("-","m").replace(".","p")}_{str(offset).replace("-","m").replace(".","p")}')
                sphere.CreateRadiusAttr(.006);pose(sphere,(x,y+offset,sim.config.table_height+.002))
                UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(mats['charcoal'])
    body=sim.assembly.objects['blender']['body'];parent=f'/Lab/body_{body}/finish'
    box('control_panel',(-.1111,0,.12),(.001,.147,.095),'rubber',parent,bevel=.004)
    # Controls face the operator; the appliance remains powered off.
    for i,y in enumerate((-.048,.048)):
        box(f'rocker_{i}',(-.114,y,.12),(.006,.018,.032),'off_switch' if i==0 else 'charcoal',parent,bevel=.002)
    knob=UsdGeom.Cylinder.Define(stage,parent+'/speed_dial');knob.CreateRadiusAttr(.016);knob.CreateHeightAttr(.009)
    knob.CreateAxisAttr('X');pose(knob,(-.116,0,.12));UsdShade.MaterialBindingAPI.Apply(knob.GetPrim()).Bind(mats['charcoal'])
    for i in range(10):box(f'motor_vent_{i}',(-.1112,-.074+i*.016,.035),(.0005,.006,.020),'rubber',parent)
    # Jug molded ribs and embossed fill marks follow its physical cylindrical envelope.
    if sim.config.scene_revision==0:
        for i in range(6):
            angle=2*np.pi*i/6
            box(f'jug_rib_{i}',(.079*np.cos(angle),.079*np.sin(angle),.335),(.003,.003,.19),'jug_frosted',parent)
    for i in range(5):
        z=.267+i*.031;r=.0801 if sim.config.scene_revision==0 else .051+(.080-.051)*(z-.228)/.216+.0001
        box(f'jug_measure_{i}',(-r,0,z),(.0004,.027,.001),'charcoal',parent)
    # Ceiling luminous fixtures occupy the same positions as the area lights.
    for i,p in enumerate(((-.3,-.7,2.8),(.4,.8,2.8),(-1.,0,2.2))):
        box(f'light_housing_{i}',(p[0],p[1],p[2]+.018),(.63,1.23,.035),'pan_gray',bevel=.004)
    if sim.config.scene_revision>=2 and os.environ.get('WIPE_FIXTURE_FINISH')=='1':
        fixture_finish(stage,box,material,mats)


def fixture_finish(stage,box,material,mats):
    """Sub-2mm trim on closed off-task enclosures; no new usable appliance.

    Front seams, lock escutcheons and folded sheet edges are authored details,
    not copied product CAD. Existing closed collision envelopes are unchanged.
    No hanging paper, opening door or active dispenser is visually promised.
    """
    from .materials import state_surface
    material('satin_fixture',(.48,.50,.51),.30,.92)
    state_surface(stage,mats['satin_fixture'],((.48,.50,.51),.30,.92))
    material('panel_seam',(.07,.08,.08),.60,.35)
    def disc(name,center,radius,mat):
        g=UsdGeom.Cylinder.Define(stage,'/Lab/finish/'+Tf.MakeValidIdentifier(name))
        g.CreateAxisAttr('X');g.CreateRadiusAttr(radius);g.CreateHeightAttr(.0006)
        pose(g,center);UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[mat])
    # Closed paper cabinet: folded front panel, fine edge seam, keyed latch.
    box('paper_cover_seam',(1.2435,1.48,1.43),(.001,.272,.312),'panel_seam',bevel=.001)
    box('paper_cover',(1.2428,1.48,1.43),(.0012,.268,.308),'satin_fixture',bevel=.0005)
    disc('paper_latch',(1.2418,1.48,1.556),.007,'brushed_trim')
    box('paper_keyway',(1.2413,1.48,1.556),(.0003,.0014,.007),'rubber',bevel=.0001)
    # The actual collision-bearing outlet remains fully visible, no fake paper.
    box('paper_outlet_fold',(1.2397,1.48,1.306),(.0006,.218,.002),'satin_fixture')
    # Soap reservoir lid joint, shallow rim and service key. Button/window are
    # the pre-existing physical pieces; none of this trim obscures them.
    box('soap_lid_joint',(1.3017,.80,1.382),(.0005,.112,.0012),'panel_seam',bevel=.0001)
    for sign in (-1,1):
        box('soap_edge_'+str(sign),(1.3017,.80+sign*.055,1.292),(.0005,.001,.174),'satin_fixture',bevel=.0001)
    disc('soap_service_lock',(1.3010,.80,1.361),.0045,'brushed_trim')
    box('soap_keyway',(1.3005,.80,1.361),(.0003,.001,.004),'rubber',bevel=.0001)
    # Cabinet perimeter reveals and screw heads give the existing closed doors
    # construction detail without introducing a manipulable decorative handle.
    for i in range(4):
        y=-1.23+i*.45
        for sign in (-1,1):
            box(f'door_reveal_{i}_{sign}',(.8821,y+sign*.213,.48),(.0005,.0015,.706),'panel_seam',bevel=.0001)
        for z in (.19,.71):
            disc(f'door_hinge_{i}_{z}',(.8818,y+.200,z),.003,'satin_fixture')
