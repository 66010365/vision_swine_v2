"""Read open source designs; all persistent geometry goes in a new, unsaved direct design."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

import adsk.core as c
import adsk.fusion as f

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/'Documentation'
REPORT = DOCS/'full-assembly-v1-report.json'
TITLE = 'VISION_SWINE_Full_Assembly_V1'
NAMES = dict(adapter='Adapter_V1', clamp='Beam_Clamp_V1', camera='Camera_Mount_V3',
             case='Jetson_Case_V5S', layout='Mounting_Layout_V1', plate='Mounting_Plate_V3')
TOL = .01


def helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'Scripts'/name/(name+'.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read(name):
    return json.loads((DOCS/name).read_bytes())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def references(r):
    recovery_path=DOCS/'jetson-reference-recovery-report.json'
    rec=read(recovery_path.name)
    v3=read('mounting-plate-v3-report.json')
    step=ROOT/'Reference'/'Jetson_Nano_Dev_Kit_3D_b01.stp'
    if (v3['status']!='cad_candidate_hardware_review_required' or not v3['single_connected_solid']
            or v3['adapter_xy_height_mm']!=[97,45,47] or digest(recovery_path)!=v3['recovery_sha256']
            or rec['assembly_inspection']!='completed_for_reported_scope' or rec['completed_trials']!=12
            or rec['replacement_map']['replaced_by_exact_path']!=923 or not rec['registration']['bounds_check']['passed']
            or digest(step)!=rec['step']['sha256']):
        raise ValueError('V3, recovery and board STEP must match the verified central design.')
    for key,row in rec['sources'].items():
        if digest(DOCS/(key+'-report.json'))!=row['sha256']:
            raise ValueError('Changed source report: '+key)
    station,=[s for s in v3['stations_from_registered_recovery']['equipment_envelope_midpoint']
              if s['beam_height_mm']==50]
    if not station['clear_in_planned_sequence']:
        raise ValueError('Central 50 mm station is blocked.')
    r.update(recovery_sha256=v3['recovery_sha256'],v3_report_sha256=digest(DOCS/'mounting-plate-v3-report.json'),
             board_step_sha256=rec['step']['sha256'],registered_station=station,assembly_sequence=v3['sequence'])
    return rec,v3,step


def sources(app,r):
    result={}
    r['open_documents']=[d.name for d in app.documents]
    for key,suffix in NAMES.items():
        name='VISION_SWINE_'+suffix
        docs=[d for d in app.documents if '_'.join(d.name.split()).casefold()==name.casefold()]
        if len(docs)!=1:
            raise ValueError(f'Open exactly one {name} document; found {len(docs)}.')
        design=f.Design.cast(docs[0].products.itemByProductType('DesignProductType'))
        if not design:
            raise ValueError('No Design: '+name)
        result[key]=design.rootComponent
    return result


def identity():
    return c.Matrix3D.create()


def matrix(values):
    m=identity()
    if not m.setWithArray(values):
        raise ValueError('Invalid registration matrix')
    return m


def shift(x,y,z):
    m=identity()
    m.translation=c.Vector3D.create(x/10,y/10,z/10)
    return m


def point(x,y,z):
    return c.Point3D.create(x/10,y/10,z/10)


def component(parent,name,pose):
    occ=parent.occurrences.addNewComponent(pose)
    occ.component.name=name
    return occ


def only(parent,name):
    matches=[o for o in parent.occurrences if o.component.name==name]
    if len(matches)!=1:
        raise ValueError('Expected one source component '+name)
    return matches[0]


def boolean(body,tool,op):
    if not f.TemporaryBRepManager.get().booleanOperation(body,tool,op):
        raise ValueError('Temporary Boolean failed')
    return body


def cylinder(x,y,bottom,top,d):
    return f.TemporaryBRepManager.get().createCylinderOrCone(point(x,y,bottom),d/20,point(x,y,top),d/20)


CUT=f.BooleanTypes.DifferenceBooleanType
UNION=f.BooleanTypes.UnionBooleanType


def add_body(comp,body,name):
    stored=comp.bRepBodies.add(body)
    if not stored or not stored.isSolid or stored.lumps.count!=1:
        raise ValueError('Cannot add connected solid: '+name)
    stored.name=name
    return stored


def copy_part(parent,source,edit_lower=False):
    if source.component.occurrences.count:
        raise ValueError('Nested source needs explicit placement: '+source.fullPathName)
    target=component(parent,source.component.name,source.transform2)
    for original in source.component.bRepBodies:
        copy=f.TemporaryBRepManager.get().copy(original)
        if edit_lower:
            if max(abs(a-b) for a,b in zip(source.transform2.asArray(),identity().asArray()))>1e-8:
                raise ValueError('Unexpected local frame of the lower plate')
            for x in (-7,7):
                boolean(copy,cylinder(x,0,-9,1,4.5),CUT)
                boolean(copy,cylinder(x,0,-4.5,1,8),CUT)
        add_body(target.component,copy,original.name)
    return target


def ring(x,y,z0,z1,od,bore):
    return boolean(cylinder(x,y,z0,z1,od),cylinder(x,y,z0-1,z1+1,bore),CUT)


def screw(x,y,tip,seat,d):
    return boolean(cylinder(x,y,tip,seat,4),cylinder(x,y,seat,seat+4,d),UNION)


def hex_nut(x,y,z0,z1,envelope_od):
    # AF7 M4 reference, clipped from the recovered Ø8.2 nut envelope.
    body=cylinder(x,y,z0,z1,envelope_od)
    manager=f.TemporaryBRepManager.get()
    for i in range(6):
        angle=i*math.pi/3
        dx,dy=math.cos(angle),math.sin(angle)
        centre=point(x+8.5*dx,y+8.5*dy,(z0+z1)/2)
        box=c.OrientedBoundingBox3D.create(centre,c.Vector3D.create(dx,dy,0),
            c.Vector3D.create(-dy,dx,0),1,2,(z1-z0+2)/10)
        boolean(body,manager.createBox(box),CUT)
    return boolean(body,cylinder(x,y,z0-1,z1+1,4.3),CUT)


def hardware(group,plate2,r):
    # Smooth reference hardware. Thread engagement represented by a nominal Ø4 bore.
    def put(b,name):
        add_body(group,b,name)
    for y in (20,70):
        put(ring(97,y,2,6,5.6,4),'M4 flush insert '+str(y))
        put(ring(97,y,12,13,9,4.3),'Flange washer '+str(y))
        put(screw(97,y,3,13,7),'Flange M4x10 '+str(y))
    for x in (90,104):
        put(ring(x,45,43,53,5.6,4),'Adapter upper insert '+str(x))
        put(screw(x,45,44.5,56.5,7),'Clamp M4x12 '+str(x))
    nominal,=[s for s in plate2['stations'] if s['offset_y_mm']==0]
    for i,h in enumerate(plate2['bores'],1):
        x,y,_=h['center_xyz_mm']
        seats={v['envelope']:v for v in nominal['hardware'] if v['hole']==i}
        for key in ('upper washer','lower washer'):
            v=seats[key]
            put(ring(x,y,*v['z_mm'],v['od_mm'],4.3),f'M4 {key} {i}')
        nut=seats['lower nut envelope']
        put(hex_nut(x,y,*nut['z_mm'],nut['od_mm']),f'M4 lower nut {i}')
        seat=seats['upper head']['z_mm'][0]
        tip=seats['lower nut envelope']['z_mm'][0]
        put(screw(x,y,tip,seat,8),f'Equipment M4 reference {i}')
    r['hardware_representation']='Smooth screws, washers and insert envelopes; equipment nuts AF7 within verified OD. Equipment screw tips flush with nuts for CAD reference. M8 rods, washers and nuts copied from Beam Clamp V1.'


def verify_bore(body,x,y,d,z0,z1):
    for face in body.faces:
        cyl=c.Cylinder.cast(face.geometry)
        if cyl and abs(cyl.axis.z)>.99999:
            bb=face.boundingBox
            actual=[cyl.origin.x*10,cyl.origin.y*10,cyl.radius*20,bb.minPoint.z*10,bb.maxPoint.z*10]
            if math.dist(actual,[x,y,d,z0,z1])<TOL:
                return actual
    raise ValueError(f'Missing bore at {(x,y)}: diameter {d}, Z={z0}..{z1}')


def require_bounds(layout,app,body,expected):
    actual=layout.box(app,body)
    if any(math.dist(p,q)>TOL for p,q in zip(actual,expected)):
        raise ValueError(f'Bounds mismatch: {actual}, expected {expected}')
    return actual


def tool_access(app,layout,root,hardware_bodies,obstacles,r):
    manager=f.TemporaryBRepManager.get()
    rows=r['live_tool_access']=[]
    def probe(label,x,y,z0,z1,diameter,targets):
        env=cylinder(x,y,z0,z1,diameter)
        radius=diameter/2
        bb=[[x-radius,y-radius,z0],[x+radius,y+radius,z1]]
        hits=[]
        for body in targets:
            if layout.overlaps(bb,layout.box(app,body)):
                overlap=boolean(manager.copy(env),body,f.BooleanTypes.IntersectionBooleanType)
                if overlap.lumps.count and overlap.volume*1000>.001:
                    hits.append(body.assemblyContext.fullPathName+'/'+body.name)
        rows.append(dict(tool=label,xy_mm=[x,y],z_mm=[z0,z1],diameter_mm=diameter,hits=hits))
        return hits
    for y in (20,70):
        if probe('Flange driver after assembly',97,y,17,47,10,obstacles+hardware_bodies):
            raise ValueError('Blocked flange driver')
    for x in (90,104):
        if probe('Upper screw key before setting beam',x,45,60.5,96,3,
                 [layout.find(root,'Lower_plate').bRepBodies.item(0),*hardware_bodies]):
            raise ValueError('Blocked recessed screw key before mounting clamp')
    # The baseline already tested the clamp and equipment at this station.
    # Recheck the tool against fasteners now present during clamp tightening.
    upper_fasteners=[b for b in hardware_bodies if b.name.startswith(('Clamp M4','Adapter upper insert'))]
    for i in range(1,5):
        rod=layout.find(root,f'Rod_{i}_ref').bRepBodies.item(0)
        x=97+(-40 if i<=2 else 40)
        y=45+(-11 if i%2 else 11)
        require_bounds(layout,app,rod,[[x-4,y-4,21],[x+4,y+4,151]])
        for side in ('lower','upper'):
            nut=layout.find(root,f'Nut_{side}_{i}_ref').bRepBodies.item(0)
            lo,hi=layout.box(app,nut)
            z0,z1=(-20,hi[2]) if side=='lower' else (lo[2],191)
            if probe('M8 pass-through before equipment '+side,x,y,z0,z1,20,upper_fasteners):
                raise ValueError('New upper fastener obstructs M8 approach')


def checks(app,root,groups,layout,rec,v3,r):
    plate=layout.find(root,'Mounting_Plate_V3').bRepBodies.item(0)
    adapter=layout.find(root,'Adapter_V1').bRepBodies.item(0)
    lower=layout.find(root,'Lower_plate').bRepBodies.item(0)
    beam=layout.find(root,'Beam_reference').bRepBodies.item(0)
    require_bounds(layout,app,plate,v3['bounds_mm'])
    require_bounds(layout,app,adapter,[[85,10,6],[109,80,53]])
    require_bounds(layout,app,lower,[[42,25,53],[152,65,61]])
    require_bounds(layout,app,beam,[[72,-55,61],[122,145,111]])
    if abs(plate.volume*1000-v3['volume_mm3'])>.1:
        raise ValueError('Plate solid volume differs from V3')
    r['verified_bores_xy_d_z_mm']=bores=[]
    for x,y in v3['original_through_bores_xy_mm']:
        bores.append(verify_bore(plate,x,y,4.5,0,6))
    case=layout.find(root,layout.CASE)
    bracket=layout.find(root,layout.BRACKET)
    mates=[]
    for h in read('jetson-case-v5-report.json')['mounting_plate_interface']['holes']:
        mates.append(layout.bore_axis(case,h['center_pcb_xy_mm'],(0,1),4.5,-8,2))
    for slot in read('camera-mount-v3-report.json')['mounting_plate_slots']:
        ends=[layout.bore_axis(bracket,xy,(0,2),4.5,0,1) for xy in slot['end_arc_centers_xz_mm']]
        mates.append([sum(coords)/2 for coords in zip(*ends)])
    r['equipment_axis_errors_mm']=[math.dist(a[:2],b) for a,b in zip(mates,v3['original_through_bores_xy_mm'])]
    if len(mates)!=6 or max(r['equipment_axis_errors_mm'])>TOL:
        raise ValueError('Equipment axes do not match V3.')
    for y in (20,70):
        bores.extend([verify_bore(plate,97,y,5.6,1.5,6),verify_bore(adapter,97,y,4.5,6,12)])
    for x in (90,104):
        bores.extend([verify_bore(lower,x,45,8,56.5,61),verify_bore(lower,x,45,4.5,53,56.5),
                      verify_bore(adapter,x,45,5.6,43,53)])
    camera=layout.find(root,layout.CAMERA)
    lens=camera.component.bRepBodies.itemByName('Lens')
    caps=sorted([(face.pointOnFace.y,face.createForAssemblyContext(camera).centroid) for face in lens.faces
                 if c.Plane.cast(face.geometry) and abs(face.geometry.normal.y)>.99999],key=lambda item:item[0])
    rear,front=[p for _,p in caps]
    angle=math.degrees(rear.vectorTo(front).angleTo(layout.V(0,0,-1)))
    r['lens']=dict(front_mm=[v*10 for v in front.asArray()],degrees_from_down=angle)
    if angle>.01 or math.dist(r['lens']['front_mm'],v3['lens']['front_xyz_mm'])>TOL:
        raise ValueError('Lens pose differs from the V3 inspection')
    usb=camera.bRepBodies.itemByName('USB cable')
    hardware_bodies=list(groups['hardware'].bRepBodies)
    clamp_bodies=layout.solids(groups['clamp'])
    r['minimum_M8_to_plate_or_adapter_mm']=min(app.measureManager.measureMinimumDistance(b,p).value*10
        for b in clamp_bodies if b.assemblyContext.component.name.startswith(('Rod_','Nut_'))
        for p in (adapter,plate))
    if abs(r['minimum_M8_to_plate_or_adapter_mm']-r['registered_station']['minimum_rod_or_nut_gap_mm'])>TOL:
        raise ValueError('M8 clearance differs from the registered 50 mm station')
    r['usb_clearance_mm']=min(app.measureManager.measureMinimumDistance(usb,b).value*10
                              for b in [adapter,*clamp_bodies,*hardware_bodies])
    obstacles=[plate,adapter,*clamp_bodies,*layout.solids(groups['camera'])]
    obstacles.extend(layout.find(root,n).bRepBodies.item(0) for n in (layout.CASE,layout.PANEL,layout.LID))
    pairs=[(a,b) for i,a in enumerate(hardware_bodies) for b in obstacles+hardware_bodies[i+1:]]
    equipment=[plate,*layout.solids(groups['camera'])]
    equipment.extend(layout.find(root,n).bRepBodies.item(0) for n in (layout.CASE,layout.PANEL,layout.LID))
    pairs.extend((a,b) for a in [adapter,*clamp_bodies] for b in equipment)
    pairs.extend((adapter,b) for b in clamp_bodies)
    boxes={}
    def box(b):
        key=(b.assemblyContext.fullPathName,b.name)
        if key not in boxes:
            boxes[key]=layout.box(app,b)
        return boxes[key]
    hits=r['interference_hits']=[]
    manager=f.TemporaryBRepManager.get()
    for a,b in pairs:
        if layout.overlaps(box(a),box(b)):
            world=manager.copy(a.nativeObject)
            if not manager.transform(world,a.assemblyContext.transform2):
                raise ValueError('Cannot position interference operand '+a.name)
            overlap=boolean(world,b,f.BooleanTypes.IntersectionBooleanType)
            if overlap.lumps.count and overlap.volume*1000>.001:
                hits.append([a.assemblyContext.fullPathName+'/'+a.name,b.assemblyContext.fullPathName+'/'+b.name])
    board_box=rec['registration']['board_bounds_plate_frame_mm']
    for b in hardware_bodies:
        if layout.overlaps(box(b),board_box):
            raise ValueError('New fastener enters recovered board envelope; review '+b.name)
    tool_access(app,layout,root,hardware_bodies,obstacles,r)
    r['scope']='Live hardware, plate, adapter and clamp interfaces; internal M8 clamp checks and source equipment contacts inherited from the verified 50 mm station. New hardware checked against recovered board bounds.'
    if hits or r['usb_clearance_mm']<=0:
        raise ValueError('Assembly clearance requires review')


def build(app,r):
    rec,v3,step=references(r)
    src=sources(app,r)
    layout=helper('CreateMountingLayoutV1')
    parts=helper('CreateMountingPlateV1').layout_parts(src['layout'],layout)
    baseline=read('mounting-layout-v1-report.json')
    for key,part in parts.items():
        if max(abs(v-w) for v,w in zip(part.transform2.asArray(),baseline['sources'][key]['transform']))>1e-6:
            raise ValueError('Layout pose differs: '+key)
    doc=app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
    doc.name=TITLE
    r['new_document_created']=True
    design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent!=f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent=f.DesignIntentTypes.HybridDesignIntentType
    design.designType=f.DesignTypes.DirectDesignType
    root=design.rootComponent
    groups={}
    for key,pose in [('plate',identity()),('adapter',shift(97,45,6)),('clamp',shift(97,45,61)),
                     ('case',parts['case'].transform2),('camera',parts['camera'].transform2)]:
        group=groups[key]=component(root,NAMES[key]+' assembly',pose)
        if key=='plate':
            members=[only(src[key],'Mounting_Plate_V3')]
        elif key=='adapter':
            members=[only(src[key],'Adapter_V1')]
        elif key=='case':
            members=[only(src[key],n) for n in (layout.CASE,layout.PANEL,layout.LID)]
        else:
            members=list(src[key].occurrences)
        if key=='clamp':
            expected={'Upper_plate','Lower_plate','Beam_reference'}
            for i in range(1,5):
                expected.add(f'Rod_{i}_ref')
                expected.update(f'{p}_{s}_{i}_ref' for s in ('lower','upper') for p in ('Nut','Washer'))
            if sorted(o.component.name for o in members)!=sorted(expected):
                raise ValueError('Clamp assembly differs from recovered inventory.')
        for member in members:
            copy_part(group.component,member,key=='clamp' and member.component.name=='Lower_plate')
    board=component(root,'Jetson Nano B01 registered STEP',matrix(rec['registration']['matrix_translation_cm']))
    manager=app.importManager
    if not manager.importToTarget(manager.createSTEPImportOptions(str(step)),board.component):
        raise ValueError('Failed to insert verified STEP into registered board component')
    # Validate one carrier datum after import, without re-running the 923-solid diagnostic.
    pcb_key=layout.read('fusion-inspection.json')['pcb_occurrence']
    carrier,=[o for o in layout.branches(board) if o.fullPathName.endswith('+'+pcb_key)]
    pcb=carrier.bRepBodies.itemByName('Body1')
    expected=rec['fresh_step_inventory']['bodies'][pcb_key+'/Body1']['bounds_step_mm']
    transformed=helper('CreateAdapterV1').root_box_from_local(expected,board.transform2.asArray())
    r['registered_pcb_bounds_mm']=require_bounds(layout,app,pcb,transformed)
    groups['hardware']=component(root,'M4 fasteners and metal inserts (reference)',identity())
    hardware(groups['hardware'].component,read('mounting-plate-v2-report.json'),r)
    r['central_mount']=dict(xy_mm=[97,45],adapter_bottom_top_mm=[6,53],beam_seat_z_mm=61)
    checks(app,root,groups,layout,rec,v3,r)
    r['status']='cad_prototype_review_required'


def run(context):
    app=c.Application.get()
    r=dict(status='started',document=TITLE,units='mm',approved_for_print=False,approved_for_installation=False,
           strength_tested=False)
    try:
        build(app,r)
    except Exception:
        r.update(status='incomplete',error=traceback.format_exc())
    if r.get('new_document_created') and app.activeDocument.name==TITLE:
        try:
            view=app.activeViewport
            cam=view.camera
            cam.viewOrientation=c.ViewOrientations.IsoTopRightViewOrientation
            cam.isFitView=True
            view.camera=cam
            view.fit()
        except Exception:
            r['view_error']=traceback.format_exc()
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=DOCS,delete=False) as out:
        json.dump(r,out,indent=2,allow_nan=False)
    os.replace(out.name,REPORT)
    app.userInterface.messageBox(r['status']+'\n'+str(REPORT))
