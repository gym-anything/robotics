"""Authored event-service alcove with collision-bearing fixed fixtures.

Dimensions/placement are a plausible design, not a scan of a named venue.
Door, trolley brakes and electrical fixtures are assembled/inactive context.
"""
import math
def build(a):
    a.box('back_wall',(1.70,0,1.50),(.075,2.50,1.50),'wall')
    a.box('left_wall',(-.75,-2.45,1.50),(2.45,.075,1.50),'wall')
    a.box('right_wall',(-.75,2.45,1.50),(2.45,.075,1.50),'wall')
    a.box('ceiling',(-.75,0,3.045),(2.45,2.50,.045),'ceiling')
    a.box('back_cove',(1.612,0,.060),(.016,2.40,.060),'trim')
    for y in (-2.364,2.364):
        a.box('side_cove_'+str(y),(-.75,y,.060),(2.35,.016,.060),'trim')
        a.box('wall_protection_'+str(y),(-.25,y,.75),(1.8,.010,.36),'counter')
    # Closed insulated service door with visible frame, kick plate and handle.
    a.box('door_panel',(1.603,1.36,1.055),(.022,.455,1.045),'door')
    for y in (.888,1.832):a.box('door_jamb_'+str(y),(1.58,y,1.075),(.034,.018,1.075),'steel')
    a.box('door_lintel',(1.58,1.36,2.14),(.034,.49,.024),'steel')
    a.box('door_kick_plate',(1.578,1.36,.21),(.003,.43,.18),'steel')
    a.tube('door_handle',(1.548,1.67,.99),(1.548,1.49,.99),.010,-1)
    a.tube('door_handle_standoff',(1.55,1.67,.99),(1.58,1.67,.99),.012,-1)
    for z in (.30,1.0,1.8):a.box('door_hinge_'+str(z),(1.571,.91,z),(.018,.015,.055),'steel')
    # Plaque backs are physical. Typography is added to their actual faces.
    for name,y in (('recycling_sign',-.30),('landfill_sign',.45)):
        a.box(name,(1.610,y,1.78),(.004,.29,.16),'sign_white')
    # Counter apron and lower shelf follow the same real supporting legs.
    a.box('counter_front_apron',(-.697,0,.79),(.003,.65,.06),'counter')
    a.box('counter_lower_shelf',(-.38,0,.18),(.29,.62,.002),'counter')
    for arm,y in (('left',-.65),('right',.65)):
        for x in (0.,.24):
            for offset in (-.12,.12):a.box(arm+'_mount_bolt_'+str(x)+'_'+str(offset),(x,y+offset,1.022),(.008,.008,.004),'steel')
        # Power/service conduits run to the column; never span the task opening.
        a.tube(arm+'_conduit_lower',(.12,y,.08),(.12,y,.93),.010,-1,'rubber')
        a.box(arm+'_controller',(.12,y,.63),(.13,.135,.17),'charcoal')
    # Braked off-task trolley with open shelves and visible caster envelopes.
    for z in (.22,.78):a.box('trolley_shelf_'+str(z),(.72,-1.64,z),(.33,.25,.018),'gray_plastic')
    for x in (.43,1.01):
        for y in (-1.85,-1.43):
            a.tube('trolley_post_'+str((x,y)),(x,y,.13),(x,y,.92),.014,-1)
            a.box('braked_caster_'+str((x,y)),(x,y,.085),(.022,.040,.065),'rubber')
    a.tube('trolley_handle',(.43,-1.85,.92),(1.01,-1.85,.92),.014,-1)
    for i,xyz in enumerate(((-.6,-.8,2.977),(.7,.6,2.977),(-1.6,.6,2.977))):
        # Back plate and perimeter frame, not an opaque block surrounding
        # the emitting face (the earlier block occluded its own light).
        a.box('luminaire_back_'+str(i),(xyz[0],xyz[1],2.993),(.32,.62,.006),'steel')
        for sign in (-1,1):
            a.box(f'luminaire_long_{i}_{sign}',(xyz[0]+sign*.31,xyz[1],2.976),(.01,.62,.011),'steel')
            a.box(f'luminaire_short_{i}_{sign}',(xyz[0],xyz[1]+sign*.61,2.976),(.30,.01,.011),'steel')
        a.box('luminaire_diffuser_'+str(i),xyz,(.30,.60,.006),'diffuser')
