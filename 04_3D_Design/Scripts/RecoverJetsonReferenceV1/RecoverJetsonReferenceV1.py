"""Fresh STEP recovery. Lengths in reports are mm; Fusion matrices translate in cm."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

import adsk
import adsk.core as c
import adsk.fusion as f

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/'Documentation'
STEP = ROOT/'Reference'/'Jetson_Nano_Dev_Kit_3D_b01.stp'
REPORT = DOCS/'jetson-reference-recovery-report.json'
TOL = .01


def adapter_tools():
    path = ROOT/'Scripts/CreateAdapterV1/CreateAdapterV1.py'
    spec = importlib.util.spec_from_file_location('vision_adapter_recovery',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(report):
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=DOCS,delete=False,suffix='.tmp') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    os.replace(stream.name,REPORT)


def xyz(point):
    return [10*n for n in point.asArray()]


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def unit(v):
    n=math.sqrt(dot(v,v))
    if n<1e-10:
        raise ValueError('Degenerate registration landmark')
    return [x/n for x in v]


def cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def axes(points):
    x=unit([b-a for a,b in zip(points[0],points[1])])
    y=[b-a for a,b in zip(points[0],points[2])]
    z=unit(cross(x,y))
    return [x,cross(z,x),z]


def registration(source,target):
    a,b=axes(source),axes(target)
    rot=[[sum(b[k][i]*a[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    matrix=[]
    for i,row in enumerate(rot):
        matrix.extend([*row,(target[0][i]-dot(row,source[0]))/10])
    return matrix+[0,0,0,1]


def transform_point(m,p):
    return [dot(m[4*i:4*i+3],p)+10*m[4*i+3] for i in range(3)]


def material(adapter,body):
    data=adapter.geometry_evidence(body)
    if adapter.body_kind(data)!='measurable_solid' or not math.isfinite(data['volume_mm3']):
        raise ValueError('Unmeasurable BRep: '+str(data))
    return data


def census(app,layout,adapter,root,report,diagnostic,scratch):
    inventory=report['fresh_step_inventory']=dict(bodies={},issues=[],totals=dict(
        bodies=0,solids=0,faces=0,edges=0,vertices=0,volume_mm3=0.0))
    items={}
    source_bodies={}
    thermal_keys={path+'/Body1' for path in diagnostic.TARGETS}
    verified=set()
    for node in [root,*layout.branches(root)]:
        prefix=node.fullPathName if f.Occurrence.cast(node) else ''
        for body in node.bRepBodies:
            key=(prefix+'/'+body.name).lstrip('/')
            total=inventory['totals']
            total['bodies']+=1
            if key in inventory['bodies']:
                raise ValueError('Ambiguous STEP occurrence/body: '+key)
            row=inventory['bodies'][key]={}
            try:
                native=body.nativeObject if body.assemblyContext else body
                data=row['native']=adapter.geometry_evidence(native)
                proxy=row['proxy']=adapter.geometry_evidence(body)
                if (adapter.body_kind(data)!='measurable_solid' or adapter.body_kind(proxy)!='measurable_solid'
                        or any(proxy[k]!=data[k] for k in ('faces','edges','vertices'))
                        or not math.isclose(data['volume_mm3'],proxy['volume_mm3'],rel_tol=1e-7,abs_tol=1e-7)):
                    raise ValueError('Native/proxy topology or volume is incomplete')
                if key in thermal_keys:
                    evidence=row['live_equivalence']={}
                    checked=diagnostic.inspect(app,scratch,body.assemblyContext,evidence,report,checkpoint=save)
                    if evidence['status']!='geometry_equivalent' or not evidence['assembly_boolean_in_native_frame']['passed']:
                        raise ValueError('Local-frame Boolean equivalence failed for '+key)
                    bounds=layout.box(app,checked)
                    snap=adapter.Item('step/'+key,checked,bounds)
                    verified.add(key)
                else:
                    snap=adapter.snapshot(app,layout,body,'step/'+key,report)
                    actual=material(adapter,snap.body)
                    if (any(actual[k]!=data[k] for k in ('faces','edges','vertices')) or
                            not math.isclose(actual['volume_mm3'],data['volume_mm3'],rel_tol=1e-7,abs_tol=1e-7)):
                        raise ValueError('Temporary STEP body differs from source')
                total['solids']+=1
                for k in ('faces','edges','vertices','volume_mm3'):
                    total[k]+=data[k]
                row['bounds_step_mm']=snap.box
                items[key]=snap
                source_bodies[key]=body
            except Exception as exc:
                row['error']=str(exc)
                inventory['issues'].append(key)
    inventory['thermal_bodies_verified']=sorted(verified)
    if inventory['issues'] or len(items)!=923 or verified!=thermal_keys:
        raise ValueError(f"STEP inventory incomplete: {len(items)} validated, expected 923; see fresh_step_inventory")
    inventory['assembly_bounds_step_mm']=layout.union([s.box for s in items.values()])
    return items,source_bodies


def seat_landmarks(body,sites,diameter,level,occ=None):
    """Read the centres of circular bore rims ON the measured planar bearing face."""
    centres,normals=[],[]
    for xy in sites:
        matches=[]
        for face in body.faces:
            plane=c.Plane.cast(face.geometry)
            if not plane or abs(plane.normal.z)<.999999 or abs(face.pointOnFace.z*10-level)>TOL:
                continue
            for edge in face.edges:
                rim=c.Circle3D.cast(edge.geometry) or c.Arc3D.cast(edge.geometry)
                if not rim or abs(20*rim.radius-diameter)>TOL or math.dist(xyz(rim.center)[:2],xy)>TOL:
                    continue
                walls=[c.Cylinder.cast(neighbour.geometry) for neighbour in edge.faces]
                if not any(w and abs(w.axis.z)>.999999 and abs(w.radius*20-diameter)<TOL
                           and math.dist(xyz(w.origin)[:2],xy)<TOL for w in walls):
                    continue
                face_world=face.createForAssemblyContext(occ) if occ else face
                edge_world=edge.createForAssemblyContext(occ) if occ else edge
                circle=c.Circle3D.cast(edge_world.geometry) or c.Arc3D.cast(edge_world.geometry)
                ok,n=face_world.evaluator.getNormalAtPoint(face_world.pointOnFace)
                if not ok or not circle:
                    raise ValueError('Unmeasurable seating face or circular rim')
                matches.append((xyz(circle.center),n.asArray()))
        if not matches or any(math.dist(p,matches[0][0])>TOL or dot(n,matches[0][1])<.999999 for p,n in matches):
            raise ValueError('Cannot uniquely measure seating face and hole at '+str(xy))
        centres.append(matches[0][0])
        normals.append(matches[0][1])
    return centres,normals


def recovery_provider(app,layout,adapter,items,pcb,inspection,bottom):
    def recover(case,audit,report):
        mapping=report['replacement_map']={}
        prefix=case.fullPathName+'+'
        empty=set()
        for row in audit['unresolved']:
            if row['classification']!='empty_topology_as_read' or not row['path'].startswith(prefix):
                raise ValueError('Unaccounted case geometry: '+str(row))
            key=row['path'][len(prefix):]+'/'+row['body']
            if key in empty:
                raise ValueError('Duplicate unresolved case body: '+key)
            empty.add(key)
        mapping.update(missing_step=sorted(empty-items.keys()),extra_step=sorted(items.keys()-empty))
        if mapping['missing_step'] or mapping['extra_step']:
            raise ValueError('Fresh STEP identities do not match the unresolved case bodies')
        sites=[s['center_xy'] for s in bottom['trial_sites'] if s['eligible'] and s['post_created']]
        frame=inspection['frame']
        if len(sites)!=4 or any(math.dist(frame[k],axis)>1e-8 for k,axis in zip('xyz',[(1,0,0),(0,1,0),(0,0,1)])):
            raise ValueError('Unexpected inspected PCB datum')
        origin=frame['origin_assembly_mm']
        expected_sites=[[x+origin[0],y+origin[1]] for x,y in sites]
        src,sn=seat_landmarks(pcb,expected_sites,2.75,origin[2])
        shell=layout.find(case,layout.CASE)
        dst,dn=seat_landmarks(shell.component.bRepBodies.item(0),sites,2.75,0,shell)
        matrix=registration(src,dst)
        expected_pose=case.transform2.asArray()
        reg=report['registration']=dict(step_centres_mm=src,plate_assembly_centres_mm=dst,
            step_face_normals=sn,post_face_normals=dn,matrix_translation_cm=matrix,
            recorded_case_pose_cm=expected_pose,landmark_errors_mm=[],normal_dots=[])
        for p,q,n,normal in zip(src,dst,sn,dn):
            reg['landmark_errors_mm'].append(math.dist(transform_point(matrix,p),q))
            reg['normal_dots'].append(dot([dot(matrix[i:i+3],n) for i in (0,4,8)],normal))
        if max(reg['landmark_errors_mm'])>TOL or max(reg['normal_dots'])>-.999999:
            raise ValueError('Four hole axes or facing bearing planes disagree')
        for i in range(3):
            if (math.dist(matrix[4*i:4*i+3],expected_pose[4*i:4*i+3])>1e-6
                    or abs(matrix[4*i+3]-expected_pose[4*i+3])*10>TOL):
                raise ValueError('Landmark transform differs from flipped V5 occurrence')
        fusion_pose=c.Matrix3D.create()
        if not fusion_pose.setWithArray(matrix):
            raise ValueError('Invalid rigid registration')
        manager=f.TemporaryBRepManager.get()
        recovered=[]
        for key,item in items.items():
            body=manager.copy(item.body)
            if not manager.transform(body,fusion_pose):
                raise ValueError('Cannot transform '+key)
            bounds=layout.box(app,body)
            expected=adapter.root_box_from_local(item.box,matrix)
            if expected is None or not adapter.coordinate_check(expected,bounds)['passed']:
                raise ValueError('Registered body bounds mismatch: '+key)
            recovered.append(adapter.Item('step/'+key,body,bounds))
        reg['board_bounds_plate_frame_mm']=layout.union([s.box for s in recovered])
        reg['bounds_check']=adapter.coordinate_check(
            adapter.root_box_from_local(report['fresh_step_inventory']['assembly_bounds_step_mm'],matrix),
            reg['board_bounds_plate_frame_mm'])
        if not reg['bounds_check']['passed']:
            raise ValueError('Complete board envelope differs after registration')
        kept=[]
        for label in (layout.CASE,layout.PANEL,layout.LID):
            node=layout.find(case,label)
            if node.bRepBodies.count!=1:
                raise ValueError('Unexpected case solid count: '+label)
            body=node.bRepBodies.item(0)
            material(adapter,body)
            kept.append(adapter.snapshot(app,layout,body,'case/'+node.fullPathName+'/'+body.name,report))
        if len(recovered)+len(kept)!=audit['body_count'] or audit['counts'].get('measurable_solid')!=3:
            raise ValueError('Case inventory not completely accounted for')
        mapping.update(replaced_by_exact_path=len(recovered),retained=[s.name for s in kept])
        return kept+recovered
    return recover


def assess_positions(report):
    """Rank complete, feasible stations only. Distances are CAD clearances, not load ratings."""
    inherited=report['inherited_equipment_hits']
    fixed_hits=bool(report['board_against_plate_case_camera'] or any(inherited.values())
                    or any(report['inherited_case_internal'].values())
                    or any(report['inherited_camera_internal'].values()))
    candidates=[]
    feasible=[]
    for name,placement in report['placements'].items():
        stations=placement['trials']
        required={(ht,h) for ht in (44,47) for h in (30,50,80)}
        measured={(ht,t['beam_height_mm']) for ht in (44,47) for t in stations
                  if abs(t['adapter_height_mm']-ht)<=TOL}
        if measured!=required or len(stations)!=6:
            raise ValueError('Missing or duplicated trial at '+name)
        for height in (44,47):
            trials=[t for t in stations if abs(t['adapter_height_mm']-height)<=TOL]
            blockers=set()
            if fixed_hits:
                blockers.add('existing equipment interference')
            hardware=[]
            blocked_after=set()
            for t in trials:
                if t['cross_equipment_hits'] or t['clamp_adapter_internal_hits']:
                    blockers.add('solid interference')
                if any(t['added_parts_in_service_corridors'].values()) or t['added_parts_ahead_of_lens']:
                    blockers.add('service or view obstruction')
                if min(t['mating_area_against_undrilled_faces_mm2'].values())<=0:
                    blockers.add('missing mating contact')
                for s in t['sockets']:
                    if s['pre_plate_axial_path_hits']:
                        blockers.add('M8 socket blocked before fitting equipment')
                    if s['assembled_axial_path_hits'] or s['seated_hits']:
                        blocked_after.add((s['rod'],s['side']))
                for s in t['adapter_screws']:
                    if s['interface']=='clamp':
                        if s['head_hits_outside_recipient'] or s['driver_before_setting_beam']:
                            blockers.add('upper adapter fastener blocked')
                    elif any(s['access'].values()):
                        blockers.add('flange fastener blocked')
                hardware.extend(min(v['to_adapter_mm'],v['to_entire_plate_mm'])
                                for v in t['hardware_clearances_mm'])
            clearance=min(hardware)
            if clearance<=0:
                blockers.add('rod, nut or washer contacts adapter or plate')
            row=dict(position=name,xy_mm=placement['xy_mm'],adapter_height_mm=height,
                     blockers=sorted(blockers),minimum_hardware_clearance_mm=clearance,
                     blocked_socket_approaches_after_assembly=len(blocked_after),
                     minimum_usb_stub_clearance_mm=min(t['usb_stub_to_added_parts_mm'] for t in trials))
            candidates.append(row)
            if not blockers:
                feasible.append(row)
    if len(candidates)!=4:
        raise ValueError('Expected two distinct XY candidates at both adapter heights')
    def rank(row):
        return (row['blocked_socket_approaches_after_assembly'],-row['minimum_hardware_clearance_mm'],
                -row['minimum_usb_stub_clearance_mm'])
    preferred=[]
    if feasible:
        best=min(map(rank,feasible))
        preferred=[row for row in feasible if rank(row)==best]
    return dict(status='cad_candidate' if preferred else 'no_feasible_tested_candidate',
                candidates=candidates,preferred=preferred,
                basis='All three beam heights; planned fastener sequence, then fewest blocked installed socket approaches, greatest worst hardware clearance, then USB stub clearance. Actual CG and load capacity unmeasured.')


def run(context):
    app=c.Application.get()
    report=dict(status='checking_step',units='mm',approved_for_print=False,approved_for_installation=False,
                assembly_inspection='incomplete',completed_trials=0)
    imported=None
    scratch_doc=None
    try:
        adapter=adapter_tools()
        layout=adapter.helper('CreateMountingLayoutV1')
        inspection=layout.read('fusion-inspection.json')
        bottom=layout.read('bottom-case-v1-report.json')
        with STEP.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        report['step']=dict(path=str(STEP),sha256=digest)
        if digest!=bottom['step_sha256'] or digest!=layout.read('step-inspection.json')['sha256']:
            raise ValueError('STEP hash differs from inspected reference')
        manager=app.importManager
        imported=manager.importToNewDocument(manager.createSTEPImportOptions(str(STEP)))
        if not imported or not imported.isValid:
            raise ValueError('STEP import failed to create a temporary document')
        imported.activate()
        adsk.doEvents()
        design=f.Design.cast(imported.products.itemByProductType('DesignProductType'))
        if not design or app.activeDocument!=imported:
            raise ValueError('Cannot inspect imported design')
        report['status']='inspecting_fresh_geometry'
        pcb_key=inspection['pcb_occurrence']+'/Body1'
        report['expected_fresh_step']=dict(pcb_key=pcb_key,solid_count=923,
            inspected_pcb_mm=inspection['pcb']['size'],inspected_board_height_mm=inspection['height_including_heatsink'],
            pcb_step_entities=dict(closed_shell=1,advanced_faces=584,edge_curves=1746,vertex_points=1164),
            topology_note='STEP entity counts are reference values; Fusion may split or merge faces during import.')
        diagnostic=adapter.helper('DiagnoseJetsonReferenceV1')
        scratch_doc=app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
        scratch_design=f.Design.cast(scratch_doc.products.itemByProductType('DesignProductType'))
        scratch_design.designType=f.DesignTypes.DirectDesignType
        items,originals=census(app,layout,adapter,design.rootComponent,report,diagnostic,scratch_design.rootComponent)
        imported.activate()
        pcb=originals[pcb_key]
        lo,hi=items[pcb_key].box
        size=[b-a for a,b in zip(lo,hi)]
        volume=pcb.getPhysicalProperties(f.CalculationAccuracy.HighCalculationAccuracy).volume*1000
        bounds=report['fresh_step_inventory']['assembly_bounds_step_mm']
        height=bounds[1][2]-bounds[0][2]
        underside=lo[2]-bounds[0][2]
        reference_volume=12048.22230842  # validation property in the hash-checked STEP
        report['step_comparison']=dict(pcb_key=pcb_key,pcb_size_mm=size,expected_pcb_size_mm=inspection['pcb']['size'],
            pcb_topology=adapter.geometry_evidence(pcb),pcb_volume_mm3=volume,step_validation_volume_mm3=reference_volume,
            assembly_height_mm=height,expected_height_mm=inspection['height_including_heatsink'],
            underside_projection_mm=underside,expected_underside_projection_mm=inspection['underside_projection'],
            expected_solids_from_import_log=923)
        if (math.dist(size,inspection['pcb']['size'])>TOL or abs(height-inspection['height_including_heatsink'])>TOL
            or abs(underside-inspection['underside_projection'])>TOL
            or not math.isclose(volume,reference_volume,rel_tol=1e-5)):
            raise ValueError('Fresh board differs from STEP validation or original Fusion inspection')
        save(report)
        report['status']='checking_registered_assembly'
        recover=recovery_provider(app,layout,adapter,items,pcb,inspection,bottom)
        adapter.build(app,report,recover_board=recover)
        trials=[t for p in report['placements'].values() for t in p['trials']]
        report['completed_trials']=len(trials)
        if len(trials)!=12:
            raise ValueError('Incomplete placement matrix')
        report['adapter_position_assessment']=assess_positions(report)
        report['scope']='Fresh board vs three V5 solids, plate and camera; all twelve clamp/adapter placements. Internal contacts among unchanged imported board components are outside the installation check. USB stub is a reference; power harness and plugs unmeasured.'
        report['status']='review_required'
    except Exception:
        report.update(status='recovery_incomplete',assembly_inspection='incomplete',error=traceback.format_exc(),
                      completed_trials=sum(len(p['trials']) for p in report.get('placements',{}).values()))
    finally:
        try:
            if scratch_doc and scratch_doc.isValid:
                scratch_doc.close(False)
            if imported and imported.isValid:
                imported.close(False)
        finally:
            save(report)
    lines=report.get('error','').splitlines()
    app.userInterface.messageBox('\n'.join([report['status'],*lines[-1:],str(REPORT)]))
