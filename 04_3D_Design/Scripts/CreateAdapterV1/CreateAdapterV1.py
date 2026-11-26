"""Adapter-local Z=0 is the flange underside. Trial coordinates are mounting-plate mm."""
import hashlib
import importlib.util
import itertools
import json
import math
import os
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

import adsk.core as c
import adsk.fusion as f

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'Documentation'
REPORT = DOCS / 'adapter-v1-report.json'
SOURCES = ('jetson-case-v5', 'camera-mount-v3', 'mounting-plate-v2', 'beam-clamp-v1')
PARAMS = dict(avWidth=24, avNeck=30, avBase=70, avHeight=47, avFlange=6, avR=2,
              avBasePitch=50, avTopPitch=14, avM4Clear=4.5, avInsertPilot=5.6, avPocket=10)
TOL = .01


def helper(name):
    path = ROOT / 'Scripts' / name / (name+'.py')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Item:
    name: str
    body: object
    box: list


def coordinate_check(expected, measured):
    errors = [[v-u for u,v in zip(a,b)] for a,b in zip(expected,measured)]
    largest = max(math.dist(a,b) for a,b in zip(expected,measured))
    return dict(expected_mm=expected, measured_mm=measured, measured_minus_expected_mm=errors,
                max_point_error_mm=largest, passed=largest<=TOL)


def precise_box(body):
    box = body.preciseBoundingBox
    if box is None:
        raise ValueError('Fusion returned no precise bounding box')
    return [[10*v for v in p.asArray()] for p in (box.minPoint,box.maxPoint)]


def root_box_from_local(local, matrix):
    # Transforming eight AABB corners is exact only for an axis permutation
    # and sign changes (including the V5 half-turn); never use an enclosing
    # approximation as an equality check for an obliquely rotated body.
    rows = [matrix[4*i:4*i+3] for i in range(3)]
    indices = [max(range(3),key=lambda k:abs(row[k])) for row in rows]
    if len(set(indices))!=3 or any(abs(abs(v)-(1 if k==axis else 0))>1e-12
                                   for row,axis in zip(rows,indices) for k,v in enumerate(row)):
        return None
    corners = [[sum(v*w for v,w in zip(row,p))+10*matrix[4*i+3]
                for i,row in enumerate(rows)] for p in itertools.product(*zip(*local))]
    return [[op(col) for col in zip(*corners)] for op in (min,max)]


def geometry_evidence(body):
    data = dict(isValid=None, solid=None, faces=None, edges=None, vertices=None, volume_mm3=None)
    for key,read in [('isValid',lambda:body.isValid),('solid',lambda:body.isSolid),
                     ('faces',lambda:body.faces.count),('edges',lambda:body.edges.count),
                     ('vertices',lambda:body.vertices.count),('volume_mm3',lambda:body.volume*1000)]:
        try:
            data[key] = read()
        except (RuntimeError,AttributeError) as exc:
            data.setdefault('read_errors',{})[key] = str(exc)
        if key=='isValid' and data[key] is not True:
            break
    return data


def body_kind(data):
    if data['isValid'] is not True or 'read_errors' in data:
        return 'unreadable'
    if data['faces']==data['edges']==data['vertices']==0 and data['volume_mm3']==0:
        return 'empty_topology_as_read'
    if not data['solid'] and data['faces']>0:
        return 'surface'
    if data['solid'] and data['faces']>0 and data['volume_mm3']>0:
        return 'measurable_solid'
    return 'unresolved'


def audit_case(layout,parent,pcb_path,out):
    """Count per occurrence, without filtering on isSolid or copying geometry."""
    out.update(status='checking',body_count=0,counts={},unresolved=[],carrier=[])
    for node in [parent,*layout.branches(parent)]:
        for body in node.bRepBodies:
            path=node.fullPathName if f.Occurrence.cast(node) else parent.name
            row=dict(path=path,body=body.name)
            out['body_count']+=1
            try:
                native=body.nativeObject if body.assemblyContext else body
                row.update(native=geometry_evidence(native),proxy=geometry_evidence(body))
                kinds=[body_kind(row[key]) for key in ('native','proxy')]
                kind=kinds[0] if kinds[0]==kinds[1] else 'native_proxy_disagreement'
            except RuntimeError as exc:
                kind='unreadable'
                row['read_error']=str(exc)
            row['classification']=kind
            out['counts'][kind]=out['counts'].get(kind,0)+1
            if path==pcb_path and body.name=='Body1':
                out['carrier'].append(row)
            if kind!='measurable_solid':
                out['unresolved'].append(row)
    out['status']='measured'


def snapshot(app,layout,body,name,report,fixed=False):
    manager = f.TemporaryBRepManager.get()
    occ = body.assemblyContext
    native = body.nativeObject if occ else body
    debug = dict(body=name, occurrence=occ.fullPathName if occ else None, tolerance_mm=TOL)
    report['frame_check_in_progress'] = debug
    expected = layout.box(app,body)
    debug['expected_proxy_bounds_mm'] = expected
    if native is None:
        raise ValueError('Missing native source for '+name)
    debug['native_bounds_mm'] = layout.box(app,native)
    copied = manager.copy(native)
    if not copied:
        raise ValueError('Cannot copy native solid: '+name)
    local_bounds = layout.box(app,copied)
    debug['native_copy_bounds_mm'] = local_bounds
    # All body proxies are obtained by descending from the root. Their contextual
    # occurrence already reports the complete pose, including the case X half-turn.
    # A detached native copy has no occurrence pose of its own.
    pose = occ.transform2 if occ else c.Matrix3D.create()
    debug['local_to_root_matrix_translation_in_cm'] = pose.asArray()
    if occ and not manager.transform(copied,pose):
        raise ValueError('Cannot apply occurrence pose: '+name)
    bounds = layout.box(app,copied)
    debug['bounds'] = coordinate_check(expected,bounds)
    # Independent point check detects a flipped or symmetric solid even if its
    # min/max coordinates are unchanged. No reliance on copy vertex ordering.
    vertex_count = native.vertices.count
    samples = []
    if occ and vertex_count:
        for i in sorted({0,vertex_count//3,2*vertex_count//3,vertex_count-1}):
            vertex = native.vertices.item(i)
            point = vertex.geometry.copy()
            if not point.transformBy(pose):
                raise ValueError('Cannot transform native vertex: '+name)
            proxy = vertex.createForAssemblyContext(occ)
            xyz = [10*v for v in proxy.geometry.asArray()]
            mapped = [10*v for v in point.asArray()]
            samples.append(coordinate_check([xyz],[mapped]))
    debug['vertex_checks'] = samples
    passed = debug['bounds']['passed'] and all(s['passed'] for s in samples)
    # Retain the PCB evidence even on successful runs, plus any other failed body.
    pcb = bool(occ and occ.component.name=='142-13449-1000-B01-SKU0000' and body.name=='Body1')
    collapsed = all(a==b for a,b in zip(bounds[0],bounds[1]))
    if pcb or not passed or collapsed:
        report.setdefault('body_frame_diagnostics',[]).append(debug)
        chain = []
        node = occ
        while node:
            placement = node.nativeObject or node
            chain.append(dict(path=node.fullPathName,
                              native_parent_matrix_cm=placement.transform2.asArray(),
                              contextual_matrix_cm=node.transform2.asArray()))
            node = node.assemblyContext
        debug['occurrence_chain_root_first'] = list(reversed(chain))
        # Re-measure the old code path to distinguish a local copy from other
        # possible deviations, without using it as analysis geometry.
        try:
            legacy = layout.box(app,manager.copy(body))
            debug['old_proxy_copy_vs_root'] = coordinate_check(expected,legacy)
            debug['old_proxy_copy_vs_native'] = coordinate_check(local_bounds,legacy)
        except RuntimeError as exc:
            debug['old_proxy_copy_diagnostic_error'] = str(exc)
        evidence = debug['geometry'] = {key:geometry_evidence(item) for key,item in
                                        [('native',native),('proxy',body),('temporary',copied)]}
        for key,item in [('native',native),('proxy',body),('temporary',copied)]:
            try:
                debug[key+'_precise_bounds_mm'] = precise_box(item)
            except (RuntimeError, ValueError) as exc:
                debug[key+'_precise_error'] = str(exc)
        local = debug.get('native_precise_bounds_mm')
        root = root_box_from_local(local,pose.asArray()) if local else None
        proxy_box = debug.get('proxy_precise_bounds_mm')
        temp_box = debug.get('temporary_precise_bounds_mm')
        if root and proxy_box:
            debug['precise_native_to_proxy'] = coordinate_check(root,proxy_box)
        if root and temp_box:
            debug['precise_native_to_temporary'] = coordinate_check(root,temp_box)
        if any(body_kind(evidence[key])!='measurable_solid' for key in ('native','proxy')):
            raise ValueError('Cannot verify source solid topology and volume: '+name+
                             '. Inspect native/proxy topology in body_frame_diagnostics.')
        if root is not None:
            # Require real extents and independent agreement before using either
            # representation. A matching pair of point-like boxes is insufficient.
            source_ok = (all(b>a for a,b in zip(*root)) and proxy_box is not None
                         and debug['precise_native_to_proxy']['passed']
                         and all(s['passed'] for s in samples))
            copy_ok = (temp_box is not None and debug['precise_native_to_temporary']['passed']
                       and evidence['temporary']['faces']==evidence['native']['faces']
                       and (evidence['temporary']['volume_mm3'] or 0)>0)
            passed = source_ok and (copy_ok or fixed)
            if passed:
                copied,bounds = (copied,temp_box) if copy_ok else (body,proxy_box)
                debug['analysis_geometry'] = 'verified temporary' if copy_ok else 'verified stationary root proxy'
        elif collapsed:
            passed = False
    if not passed:
        raise ValueError('Assembly frame mismatch: '+name)
    report.pop('frame_check_in_progress')
    return Item(name,copied,bounds)


def translated(item,dx,dy,dz):
    manager = f.TemporaryBRepManager.get()
    body = manager.copy(item.body)
    matrix = c.Matrix3D.create()
    matrix.translation = c.Vector3D.create(dx/10,dy/10,dz/10)
    if not manager.transform(body,matrix):
        raise ValueError('Cannot position '+item.name)
    return Item(item.name,body,[[x+dx,y+dy,z+dz] for x,y,z in item.box])


def verify_plate_frame(plate,reference,live,report):
    checks = report['plate_frame'] = dict(bounds=coordinate_check(reference['bounds_mm'],plate.box),
                                          bores=[])
    lo,hi = [row[2] for row in plate.box]
    measured = []
    for face in plate.body.faces:
        cyl = c.Cylinder.cast(face.geometry)
        if not cyl or abs(cyl.axis.z)<.99999 or abs(20*cyl.radius-4.5)>TOL:
            continue
        bb = face.boundingBox
        if max(abs(bb.minPoint.z*10-lo),abs(bb.maxPoint.z*10-hi))>TOL:
            continue
        p = cyl.origin
        t = (lo/10-p.z)/cyl.axis.z
        measured.append([10*(p.x+t*cyl.axis.x),10*(p.y+t*cyl.axis.y),lo])
    checks['measured_plate_centres_mm'] = measured
    bores = reference['bores']
    stations = live['mounting_points']
    if len(bores)!=6 or len(measured)!=6 or len(stations)!=6:
        raise ValueError('Expected six complete plate bores and six assembly mounting centres')
    remaining = measured.copy()
    for bore,station in zip(bores,stations):
        xyz = bore['center_xyz_mm']
        nearest = min(remaining,key=lambda p:math.dist(xyz,p))
        remaining.remove(nearest)
        checks['bores'].append(dict(plate=coordinate_check([xyz],[nearest]),
            mating_part=station['part'],mating_axis=coordinate_check([xyz],[station['center_xyz_mm']])))
    if not checks['bounds']['passed'] or any(not (b['plate']['passed'] and b['mating_axis']['passed'])
                                             for b in checks['bores']):
        raise ValueError('Plate bounds or mounting centres differ from Mounting Plate V2 report')


def source_models(app,layout,refs,report,recover_board=None):
    plate_docs, clamp_docs = [], []
    v5_docs = []
    for doc in app.documents:
        design = f.Design.cast(doc.products.itemByProductType('DesignProductType'))
        if not design:
            continue
        children = {o.component.name:o for o in design.rootComponent.occurrences}
        if len(children)!=design.rootComponent.occurrences.count and 'Mounting_Plate_V2' in children:
            raise ValueError('Ambiguous duplicate components in '+doc.name)
        if 'Mounting_Plate_V2' in children:
            plate_docs.append((doc,children))
        if {'Lower_plate','Upper_plate','Beam_reference','Rod_4_ref'} <= children.keys():
            clamp_docs.append((doc,children,design))
        if {layout.CASE,layout.PANEL,layout.LID} <= children.keys():
            v5_docs.append((doc,design))
    if len(plate_docs)!=1 or len(clamp_docs)!=1:
        raise ValueError(f'Open one full Mounting Plate V2 and one Beam Clamp V1 assembly. Found {len(plate_docs)} and {len(clamp_docs)}.')
    doc,children = plate_docs[0]
    clamp_doc,clamps,clamp_design = clamp_docs[0]
    report['open_sources'] = [doc.name,clamp_doc.name]
    plate_occ = children.pop('Mounting_Plate_V2')
    excluded = children.pop('Mounting_Plate_V1',None)
    report['plate_source_selection'] = dict(document=doc.name,selected=plate_occ.fullPathName,
        excluded_v1=excluded.fullPathName if excluded else None,
        v1_light_bulb_on=excluded.isLightBulbOn if excluded else None,
        remaining_root_occurrences=[o.fullPathName for o in children.values()])
    if plate_occ.bRepBodies.count!=1 or len(children)!=1:
        raise ValueError('Expected one V2 plate solid and one complete layout')
    wrapper, = children.values()
    pv1 = helper('CreateMountingPlateV1')
    baseline = layout.read('mounting-layout-v1-report.json')
    inspection = layout.read('fusion-inspection.json')
    comparison=report['source_topology_comparison']
    comparison['v5_historical_inventory']=baseline['sources']['case']['source_inventory']
    comparison['open_standalone_v5']=[]
    for v5doc,v5design in v5_docs:
        audit=dict(document=v5doc.name)
        comparison['open_standalone_v5'].append(audit)
        audit_case(layout,v5design.rootComponent,inspection['pcb_occurrence'],audit)
    embedded=pv1.layout_parts(wrapper,layout)['case']
    audit=comparison['case_in_mounting_plate_v2']={}
    audit_case(layout,embedded,embedded.fullPathName+'+'+inspection['pcb_occurrence'],audit)
    parts = pv1.verify_layout(wrapper,layout,baseline)
    report['case_pose'] = dict(path=parts['case'].fullPathName,
        expected_matrix_cm=baseline['sources']['case']['transform'],
        measured_matrix_cm=parts['case'].transform2.asArray())
    # Diagnose the exact carrier occurrence from inspection before scanning all
    # equipment; the same verified snapshot is used later for interference.
    target_path = parts['case'].fullPathName+'+'+inspection['pcb_occurrence']
    carriers = [o for o in layout.branches(parts['case']) if o.fullPathName==target_path]
    if len(carriers)!=1:
        raise ValueError('Cannot uniquely identify inspected PCB occurrence: '+target_path)
    carrier = carriers[0].bRepBodies.itemByName('Body1')
    if carrier is None:
        raise ValueError('Cannot find Body1 in '+target_path)
    pcb_key = 'case/'+target_path+'/Body1'
    report['pcb_preflight'] = dict(path=target_path,
        inspected_board_size_mm=inspection['pcb']['size'],status='checking')
    recovered = None
    if recover_board is None:
        pcb_item = snapshot(app,layout,carrier,pcb_key,report,fixed=True)
        if audit['unresolved']:
            raise ValueError('Case contains unresolved geometry; inspect source_topology_comparison before assembly interference.')
    else:
        recovered = recover_board(parts['case'],audit,report)
    report['pcb_preflight']['status'] = 'passed' if recovered is None else 'validated_reimport'
    live = {}
    layout.inspect(app,f.Design.cast(doc.products.itemByProductType('DesignProductType')),parts,
                   refs['jetson-case-v5'],refs['camera-mount-v3'],20,live,
                   case_geometry=None if recovered is None else [item.body for item in recovered])
    equipment = [snapshot(app,layout,plate_occ.bRepBodies.item(0),'Mounting_Plate_V2',report)]
    r = refs['mounting-plate-v2']
    verify_plate_frame(equipment[0],r,live,report)
    if abs(plate_occ.bRepBodies.item(0).getPhysicalProperties(
                f.CalculationAccuracy.HighCalculationAccuracy).volume*1000-r['volume_mm3'])>.1:
        raise ValueError('Active plate differs from V2 report')
    for key,part in parts.items():
        if key=='case' and recovered is not None:
            equipment.extend(recovered)
            continue
        if recovered is not None:
            bodies=[b for node in [part,*layout.branches(part)] for b in node.bRepBodies]
            for b in bodies:
                if body_kind(geometry_evidence(b))!='measurable_solid':
                    raise ValueError('Unresolved camera geometry: '+b.assemblyContext.fullPathName+'/'+b.name)
        else:
            bodies = layout.solids(part)
        if len(bodies)!=r['interference_scope'][key+'_bodies']:
            raise ValueError('Incomplete equipment geometry: '+key)
        for b in bodies:
            label=key+'/'+b.assemblyContext.fullPathName+'/'+b.name
            equipment.append(pcb_item if label==pcb_key else snapshot(app,layout,b,label,report,fixed=True))
    p = refs['beam-clamp-v1']['parameters_mm']
    for name,val in p.items():
        actual = clamp_design.userParameters.itemByName(name)
        if not actual or (name!='bcBeamH' and abs(actual.value*10-val)>TOL):
            raise ValueError('Clamp differs from report: '+name)
    names = {'Beam_reference','Lower_plate','Upper_plate'}
    for i in range(1,5):
        names.add(f'Rod_{i}_ref')
        for side in ('lower','upper'):
            names.update((f'Nut_{side}_{i}_ref',f'Washer_{side}_{i}_ref'))
    if names != clamps.keys():
        raise ValueError('Unexpected clamp component inventory')
    fixtures = {}
    for name,occ in clamps.items():
        if occ.bRepBodies.count!=1 or not occ.bRepBodies.item(0).isSolid:
            raise ValueError('Invalid clamp reference '+name)
        if recovered is not None and body_kind(geometry_evidence(occ.bRepBodies.item(0)))!='measurable_solid':
            raise ValueError('Unresolved clamp geometry: '+name)
        fixtures[name] = snapshot(app,layout,occ.bRepBodies.item(0),name,report)
    b = fixtures['Beam_reference'].box
    if math.dist(b[0],[-25,-100,0])>TOL or math.dist(b[1][:2],[25,100])>TOL:
        raise ValueError('Expected a centred 50 mm wide beam')
    lower=fixtures['Lower_plate'].box
    if math.dist(lower[0],[-55,-20,-8])>TOL or math.dist(lower[1],[55,20,0])>TOL:
        raise ValueError('Clamp lower plate is outside the inspected coordinate frame')
    for i in range(1,5):
        rod=fixtures[f'Rod_{i}_ref'].box
        centre=[(a+b)/2 for a,b in zip(*rod)]
        expected=[-40 if i<=2 else 40,-11 if i%2 else 11,b[1][2]/2]
        if math.dist(centre,expected)>TOL or abs(rod[1][2]-rod[0][2]-130)>TOL:
            raise ValueError('Clamp rod does not match the inspected layout')
    return equipment,fixtures,live


def model(design,bc,g):
    for name,val in PARAMS.items():
        design.userParameters.add(name,g.value(f'{val} mm'),'mm','Adapter V1 prototype')
    p = bc.Part(design,g,'Adapter_V1')
    p.rect('avWidth','avBase','0','avFlange')
    sk,pts = p.sketch('avFlange',[('-avWidth/2','-avNeck/2'),('avWidth/2','avNeck/2')])
    sk.sketchCurves.sketchLines.addTwoPointRectangle(*pts)
    g.extrude(p.comp,sk,'avHeight-avFlange',bc.POS,f.FeatureOperations.JoinFeatureOperation,'Central neck')
    edges=c.ObjectCollection.create()
    for e in p.comp.bRepBodies.item(0).edges:
        if abs(e.startVertex.geometry.z-e.endVertex.geometry.z)*10>TOL:
            edges.add(e)
    if edges.count!=8:
        raise ValueError('Unexpected external corner count')
    fillets=p.comp.features.filletFeatures
    inp=fillets.createInput()
    inp.edgeSetInputs.addConstantRadiusEdgeSet(edges,g.value('avR'),False)
    fillets.add(inp).name='Rounded flange and neck corners'
    for sign in ('','-'):
        p.circle('0',sign+'avBasePitch/2','0','avM4Clear','avFlange',bc.CUT,'Lower M4 clearance')
        p.circle(sign+'avTopPitch/2','0','avHeight-avPocket','avInsertPilot','avPocket',bc.CUT,'Upper insert trial pocket')
    return p


def inspect_bores(part,height):
    measurements=[]
    for face in part.comp.bRepBodies.item(0).faces:
        cyl=c.Cylinder.cast(face.geometry)
        if not cyl or abs(cyl.axis.z)<.99999:
            continue
        diameter=cyl.radius*20
        if min(abs(diameter-d) for d in (4.5,5.6))>TOL:
            continue
        bb=face.boundingBox
        measurements.append([cyl.origin.x*10,cyl.origin.y*10,diameter,bb.minPoint.z*10,bb.maxPoint.z*10])
    expected=[[0,y,4.5,0,6] for y in (-25,25)]+[[x,0,5.6,height-10,height] for x in (-7,7)]
    if len(measurements)!=4 or any(not any(math.dist(row,m)<TOL for m in measurements) for row in expected):
        raise ValueError('Adapter hole and insert-pocket geometry did not match the design')
    return measurements


def cylinder(g,name,x,y,z0,z1,diameter):
    lo,hi=sorted((z0,z1))
    r=diameter/2
    body=f.TemporaryBRepManager.get().createCylinderOrCone(g.point(x,y,lo),r/10,g.point(x,y,hi),r/10)
    return Item(name,body,[[x-r,y-r,lo],[x+r,y+r,hi]])


def box_item(layout,g,name,bounds):
    lo,hi=bounds
    centre=g.point(*[(a+b)/2 for a,b in zip(lo,hi)])
    dims=[(b-a)/10 for a,b in zip(lo,hi)]
    body=f.TemporaryBRepManager.get().createBox(c.OrientedBoundingBox3D.create(
        centre,layout.V(1,0,0),layout.V(0,1,0),*dims))
    return Item(name,body,bounds)


def hits(layout,item,items):
    manager=f.TemporaryBRepManager.get()
    found=[]
    for other in items:
        if not layout.overlaps(item.box,other.box):
            continue
        target=manager.copy(item.body)
        if not manager.booleanOperation(target,other.body,f.BooleanTypes.IntersectionBooleanType):
            raise ValueError(f'Boolean check failed: {item.name}, {other.name}')
        if target.lumps.count and target.volume*1000>.001:
            found.append(other.name)
    return found


def distance(app,a,b):
    return app.measureManager.measureMinimumDistance(a.body,b.body).value*10


def placed_clamp(layout,g,fixtures,x,y,bottom,h):
    # Lower plate stays fixed; rods remain centred, upper hardware follows beam height.
    beam=fixtures['Beam_reference'].box
    delta=h-beam[1][2]
    origin_z=bottom-fixtures['Lower_plate'].box[0][2]
    result={}
    for name,item in fixtures.items():
        if name=='Beam_reference':
            replacement=box_item(layout,g,name,[beam[0],beam[1][:2]+[h]])
            result[name]=translated(replacement,x,y,origin_z)
        else:
            dz=delta/2 if name.startswith('Rod_') else delta if 'upper' in name.lower() else 0
            result[name]=translated(item,x,y,origin_z+dz)
    return result

def screw_checks(layout,g,x,y,zplate,adapter,clamp,equipment):
    """Only the adapter holes exist. Host materials at proposed future seats are not cut."""
    rows=[]
    plate=equipment[0]
    top=adapter.box[1][2]
    beam_seat=clamp['Lower_plate'].box[1][2]
    other_clamp=[v for k,v in clamp.items() if k not in ('Beam_reference','Lower_plate')]
    for dx in (-PARAMS['avTopPitch']/2,PARAMS['avTopPitch']/2):
        cx=x+dx
        # M4 x 12 socket head, OD7 x 4. Recess 4.5 mm from beam seat.
        seat=beam_seat-4.5
        head=cylinder(g,'recessed upper head',cx,y,seat,seat+4,7)
        driver=cylinder(g,'upper driver',cx,y,beam_seat,beam_seat+35,10)
        rows.append(dict(interface='clamp',xy_mm=[cx,y],adapter_pocket_z_mm=[top-10,top],
            proposed_clamp_clearance_diameter_mm=4.5,proposed_counterbore_diameter_depth_mm=[8,4.5],
            head_recess_below_beam_mm=beam_seat-head.box[1][2],
            screw_engagement_into_adapter_mm=top-(seat-12),
            head_hits_outside_recipient=hits(layout,head,[adapter,*equipment,*other_clamp]),
            driver_before_setting_beam=hits(layout,driver,[adapter,*equipment]),
            driver_after_setting_beam=hits(layout,driver,list(clamp.values()))))
    for dy in (-PARAMS['avBasePitch']/2,PARAMS['avBasePitch']/2):
        cy=y+dy
        flange_top=zplate+PARAMS['avFlange']
        # Washer and head above flange, washer and metal nut under equipment plate.
        envelopes=[cylinder(g,'head+washer',x,cy,flange_top,flange_top+5,9),
                   cylinder(g,'nut+washer',x,cy,plate.box[0][2]-4.2,plate.box[0][2],9),
                   cylinder(g,'driver above flange',x,cy,flange_top+5,flange_top+35,10),
                   cylinder(g,'socket from below',x,cy,plate.box[0][2]-35,plate.box[0][2]-1,12)]
        rows.append(dict(interface='mounting_plate',xy_mm=[x,cy],proposed_plate_clearance_mm=4.5,
            proposed_screw='M4 x 20, 1 mm washers both sides, 3.2 mm nut; verify actual stack',
            access={env.name:hits(layout,env,[adapter,*equipment,*clamp.values()]) for env in envelopes}))
    return rows


def evaluate(app,layout,g,fixtures,equipment,local,x,y,h,live):
    plate=equipment[0]
    zplate=plate.box[1][2]
    adapter=translated(local,x,y,zplate)
    clamp=placed_clamp(layout,g,fixtures,x,y,adapter.box[1][2],h)
    additions=[adapter,*clamp.values()]
    cross={b.name:hits(layout,b,equipment) for b in additions}
    internal={}
    for i,b in enumerate(additions):
        internal[b.name]=hits(layout,b,additions[i+1:])
    manager=f.TemporaryBRepManager.get()
    contact={}
    for label,mate in [('base',plate),('neck',clamp['Lower_plate'])]:
        overlap=manager.imprintOverlapBodies(adapter.body,mate.body,False)
        if not overlap[0]:
            raise ValueError('Cannot measure mating area: '+label)
        contact[label]=sum(face.area*100 for face in overlap[3] if c.Plane.cast(face.geometry)
                           and abs(face.geometry.normal.z)>.99999)
    sockets=[]
    hardware=[]
    for i in range(1,5):
        rod=clamp[f'Rod_{i}_ref']
        cx,cy=[sum(row[k] for row in rod.box)/2 for k in (0,1)]
        for key in [rod.name]+[f'{kind}_{side}_{i}_ref' for kind in ('Nut','Washer') for side in ('lower','upper')]:
            item=clamp[key]
            hardware.append(dict(name=key,to_adapter_mm=distance(app,item,adapter),
                                 to_entire_plate_mm=distance(app,item,plate)))
        for side,sign in [('lower',-1),('upper',1)]:
            nut=clamp[f'Nut_{side}_{i}_ref']
            end=0 if sign<0 else 1
            inner=nut.box[1-end][2]
            outer=nut.box[end][2]
            ignore={nut.name,rod.name}
            others=[adapter,*equipment,*[b for k,b in clamp.items() if k not in ignore]]
            seated=cylinder(g,'socket OD20',cx,cy,inner,outer+sign*30,20)
            # Straight axial insertion to/from outside the whole loaded assembly.
            far=(min(v.box[0][2] for v in others)-40 if sign<0 else max(v.box[1][2] for v in others)+40)
            path=cylinder(g,'axial path',cx,cy,inner,far,20)
            sockets.append(dict(rod=i,side=side,rod_tip_z_mm=rod.box[end][2],
                tip_height_above_plate_top_mm=rod.box[end][2]-zplate if sign<0 else None,
                seated_hits=hits(layout,seated,others),assembled_axial_path_hits=hits(layout,path,others),
                pre_plate_axial_path_hits=hits(layout,path,[b for b in additions if b.name not in ignore])))
    usb,=[b for b in equipment if b.name.endswith('/USB cable')]
    minimum_usb=min(distance(app,usb,b) for b in additions)
    service={}
    for s in live['service_access']['corridors']:
        corridor=box_item(layout,g,s['region'],s['bounds'])
        service[s['region']]=hits(layout,corridor,additions)
    screws=screw_checks(layout,g,x,y,zplate,adapter,clamp,equipment)
    return dict(beam_height_mm=h,adapter_height_mm=local.box[1][2]-local.box[0][2],
        cross_equipment_hits={k:v for k,v in cross.items() if v},
        clamp_adapter_internal_hits={k:v for k,v in internal.items() if v},
        mating_area_against_undrilled_faces_mm2=contact,hardware_clearances_mm=hardware,
        sockets=sockets,adapter_screws=screws,usb_stub_to_added_parts_mm=minimum_usb,
        usb_stub_lower_bound_for_camera_slot_travel_mm=max(0,minimum_usb-4),
        added_parts_in_service_corridors=service,
        added_parts_ahead_of_lens=[b.name for b in additions if b.box[0][2]<live['view_check']['lens_front_z_mm']-.01])


def build(app,report,recover_board=None):
    layout=helper('CreateMountingLayoutV1')
    bc=helper('CreateBeamClampV1')
    g=helper('CreateCameraMountV3')
    # The standalone prototype has no geometry dependency on the five source files.
    doc=app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
    doc.name='VISION SWINE Adapter V1'
    design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent!=f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent=f.DesignIntentTypes.HybridDesignIntentType
    design.designType=f.DesignTypes.ParametricDesignType
    part=model(design,bc,g)
    if not design.computeAll():
        raise ValueError('Standalone adapter regeneration failed')
    body=part.body()
    nominal=layout.box(app,body)
    sizes=[b-a for a,b in zip(*nominal)]
    if math.dist(sizes,[24,70,47])>TOL or abs(nominal[0][2])>TOL:
        raise ValueError('Nominal standalone adapter dimensions failed')
    report['standalone_adapter']=dict(status='created_and_measured',document=doc.name,
        size_mm=sizes,volume_mm3=g.volume(part.comp),single_connected_solid=True,
        measured_local_bores_xy_d_zmin_zmax_mm=inspect_bores(part,47))
    app.activeViewport.fit()
    report['assembly_inspection']='incomplete'
    step=ROOT/'Reference'/'Jetson_Nano_Dev_Kit_3D_b01.stp'
    with step.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    comparison=report['source_topology_comparison']={'step_sha256':digest}
    if digest=='054c594a8b08185f8aab22e30976f7d7df3f16771455a1485381111e068f279d':
        comparison['verified_step_pcb']=dict(brep_entity=26781,kind='MANIFOLD_SOLID_BREP',
            closed_shells=1,advanced_faces=584,edge_curves=1746,vertex_points=1164,
            file_validation_volume_mm3=12048.22230842,
            basis='Read-only STEP entity graph and validation property, matched by exact file SHA-256.')
    refs={}
    report['sources']={}
    for name in SOURCES:
        raw=(DOCS/(name+'-report.json')).read_bytes()
        refs[name]=json.loads(raw.decode('utf-8-sig'))
        report['sources'][name]=dict(status=refs[name]['status'],sha256=hashlib.sha256(raw).hexdigest())
    plate=refs['mounting-plate-v2']
    clamp_report=refs['beam-clamp-v1']
    for h in (30,50,80):
        if not any(s['beam_width_mm']==50 and s['beam_height_mm']==h and s['geometry_checks_passed']
                   for s in clamp_report['stations']):
            raise ValueError('Missing validated beam station '+str(h))
    equipment,fixtures,live=source_models(app,layout,refs,report,recover_board)
    if recover_board is not None:
        board=[item for item in equipment if item.name.startswith('step/')]
        fixed=[item for item in equipment if not item.name.startswith('step/')]
        report['board_against_plate_case_camera']={item.name:found for item in board
            if (found:=hits(layout,item,fixed))}
    report.update(parameters_mm=PARAMS,frame='Adapter-local origin at base centre. Trial coordinates use equipment assembly axes; beam along +Y.',
        lens=live['view_check'],power_and_USB_plug_scope=refs['jetson-case-v5']['physical_access'],
        camera_USB_reference=refs['camera-mount-v3']['usb_reference'],
        plate_USB_baseline_lower_bound_mm=plate['usb_gap_lower_bound_over_travel_mm'],
        inherited_equipment_hits=dict(plate_case=plate['plate_case_hits'],case_camera=live['cross_assembly_interferences'],
            plate_camera_nominal=next(s['plate_camera_hits'] for s in plate['stations'] if s['offset_y_mm']==0)),
        inherited_case_internal=refs['jetson-case-v5']['interferences'],
        inherited_camera_internal=refs['camera-mount-v3']['camera_bracket_interference'],
        inherited_camera_extreme_hardware=[r for s in plate['stations'] if s['offset_y_mm']==-4
                                          for r in s['hardware'] if r['hole']==6 and r['interfering_bodies']],
        assembly_sequence=['Trial upper pockets use metal threaded inserts; confirm insert OD, pilot and engagement.',
            'Future modification: recess upper M4 heads in lower clamp before fitting it to beam. Remove beam to access these heads again.',
            'Fit clamp plates and centre M8 rods. Tighten using pass-through socket before attaching equipment.',
            'Offer equipment plate to adapter, fit flange bolts from above and hold nuts below only where reported path is clear. Remove plate first for blocked M8 approaches.'],
        limitations=['Adapter screws and recipient holes are proposals. Mating areas measured against undrilled reference faces.',
            'Trial equipment camera is centred. Actual cable plugs, bend radii, power harness, hand clearance, ratchet swing and load testing remain open.'])
    overall=live['assembly_bounds']
    centre=[sum(row[k] for row in overall)/2 for k in (0,1)]
    candidates={'reserved':plate['beam_clamp']['candidate_xy_mm'],'equipment_envelope_midpoint':centre}
    report['placements']={}
    for key,xy in candidates.items():
        report['placements'][key]=dict(xy_mm=xy,offset_to_geometric_midpoint_mm=math.dist(xy,centre),
            offset_to_subassembly_box_centres_mm={name:math.dist(xy,[sum(row[k] for row in live[name+'_bounds'])/2
                for k in (0,1)]) for name in ('case','camera')},trials=[])
    report['moment_basis']='Geometric distances only. For measured CG, arm=hypot(CGx-x,CGy-y) and gravitational moment=m*g*arm/1000 (kg, mm, N m).'
    height=design.userParameters.itemByName('avHeight')
    try:
        for ht in (47,44):
            height.expression=f'{ht} mm'
            if not design.computeAll():
                raise ValueError('Cannot regenerate adapter height')
            body=part.body()
            local=snapshot(app,layout,body,'Adapter_V1',report)
            dims=[b-a for a,b in zip(*local.box)]
            if math.dist(dims,[24,70,ht])>TOL or abs(local.box[0][2])>TOL:
                raise ValueError('Adapter dimensions differ from trial')
            report.setdefault('measured_local_bores_xy_d_zmin_zmax_mm',{})[str(ht)]=inspect_bores(part,ht)
            for key,(x,y) in candidates.items():
                for h in (30,50,80):
                    trial=evaluate(app,layout,g,fixtures,equipment,local,x,y,h,live)
                    report['placements'][key]['trials'].append(trial)
    finally:
        height.expression='47 mm'
        if not design.computeAll():
            raise ValueError('Cannot restore prototype height')
    for placement in report['placements'].values():
        placement['height_comparison']={}
        for ht in (47,44):
            trials=[t for t in placement['trials'] if abs(t['adapter_height_mm']-ht)<TOL]
            screws=[s for t in trials for s in t['adapter_screws']]
            placement['height_comparison'][str(ht)]=dict(
                minimum_lower_tip_above_plate_mm=min(s['tip_height_above_plate_top_mm']
                    for t in trials for s in t['sockets'] if s['side']=='lower'),
                blocked_M8_after_plate=sorted({(s['rod'],s['side']) for t in trials for s in t['sockets']
                                               if s['assembled_axial_path_hits']}),
                blocked_M8_before_plate=sorted({(s['rod'],s['side']) for t in trials for s in t['sockets']
                                                if s['pre_plate_axial_path_hits']}),
                blocked_flange_fastener_access_xy_mm=sorted({tuple(s['xy_mm']) for s in screws
                    if s['interface']=='mounting_plate' and any(s['access'].values())}),
                added_parts_interference=any(t['cross_equipment_hits'] or t['clamp_adapter_internal_hits'] for t in trials))
    report.update(single_connected_solid=part.body().isSolid,volume_mm3=g.volume(part.comp),
        status='review_required',assembly_inspection='completed_for_reported_scope',final_mounting_position=None,physical_CG=None)
    app.activeViewport.fit()


def run(context):
    app=c.Application.get()
    report=dict(status='checking_sources',units='mm',script_path=str(Path(__file__).resolve()),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                approved_for_print=False,approved_for_installation=False,
                load_tested=False)
    try:
        build(app,report)
    except Exception:
        if report.get('pcb_preflight',{}).get('status')=='checking':
            report['pcb_preflight']['status']='blocked'
        report.update(status='assembly_inspection_incomplete' if 'standalone_adapter' in report else 'failed',
                      error=traceback.format_exc())
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=DOCS,delete=False,suffix='.tmp') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    os.replace(stream.name,REPORT)
    lines=report.get('error','').splitlines()
    app.userInterface.messageBox('\n'.join([report['status'],*lines[-1:],str(REPORT)]))
