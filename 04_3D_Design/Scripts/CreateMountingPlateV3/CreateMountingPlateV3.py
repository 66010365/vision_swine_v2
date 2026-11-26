"""Central adapter interface. Lengths in the report are mm; Fusion geometry is in cm."""
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

SCRIPTS = Path(__file__).resolve().parents[1]
DOCS = SCRIPTS.parent / 'Documentation'
REPORT = DOCS / 'mounting-plate-v3-report.json'
TOL = 0.01
HOLES = [(6,38), (6,28), (126,38), (126,28), (172,9), (172,29)]


def helper(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / name / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(report):
    raw = (DOCS/'jetson-reference-recovery-report.json').read_bytes()
    recovery = json.loads(raw)
    if (recovery['assembly_inspection'] != 'completed_for_reported_scope' or recovery['completed_trials'] != 12
            or recovery['fresh_step_inventory']['totals']['solids'] != 923
            or recovery['replacement_map']['replaced_by_exact_path'] != 923
            or not recovery['registration']['bounds_check']['passed'] or recovery['board_against_plate_case_camera']):
        raise ValueError('A completed, registered recovery baseline is required.')
    refs = {}
    for key, provenance in recovery['sources'].items():
        data = (DOCS/(key+'-report.json')).read_bytes()
        if hashlib.sha256(data).hexdigest() != provenance['sha256']:
            raise ValueError('Source report changed since recovery: '+key)
        refs[key] = json.loads(data)
    report['recovery_sha256'] = hashlib.sha256(raw).hexdigest()
    report['recovery_sources'] = recovery['sources']
    return recovery, refs


def check_bores(body, holes):
    for xy in holes:
        matches = []
        for face in body.faces:
            cyl = c.Cylinder.cast(face.geometry)
            if cyl and abs(cyl.axis.z)>.99999 and abs(cyl.radius*20-4.5)<TOL:
                if math.dist(xy,[cyl.origin.x*10,cyl.origin.y*10])<TOL:
                    box = face.boundingBox
                    matches.append([box.minPoint.z*10,box.maxPoint.z*10])
        if len(matches)!=1 or math.dist(matches[0],[0,6])>TOL:
            raise ValueError('Original through bore not verified at '+str(xy))


def select_source(app, recovery, report):
    active = app.activeDocument
    expected = recovery['plate_source_selection']['remaining_root_occurrences']
    documents = report['open_documents'] = []
    candidates = []
    for doc in app.documents:
        design = f.Design.cast(doc.products.itemByProductType('DesignProductType'))
        roots = list(design.rootComponent.occurrences) if design else []
        plates = [o for o in roots if o.component.name == 'Mounting_Plate_V2']
        layouts = [o for o in roots if o.fullPathName in expected]
        documents.append(dict(name=doc.name, active=doc == active, plate_count=len(plates),
                              layout_count=len(layouts), root_paths=[o.fullPathName for o in roots]))
        if len(plates) == 1 and len(layouts) == 1:
            candidates.append((doc, plates[0], layouts[0]))
    for candidate in candidates:
        if candidate[0] == active:
            return candidate
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError('Open the assembly containing Mounting_Plate_V2 and the verified layout. See open_documents.')
    raise ValueError('Multiple matching assemblies are open. Activate the intended V2 assembly and rerun.')


def live_source(app, layout, recovery, plate, report):
    doc, old, wrapper = select_source(app, recovery, report)
    report['source_document'] = doc.name
    parts = helper('CreateMountingPlateV1').layout_parts(wrapper,layout)
    baseline = layout.read('mounting-layout-v1-report.json')
    for key,part in parts.items():
        if max(abs(a-b) for a,b in zip(part.transform2.asArray(),baseline['sources'][key]['transform']))>1e-6:
            raise ValueError('Source pose changed: '+key)
    if old.bRepBodies.count!=1:
        raise ValueError('V2 plate inventory differs.')
    body = old.bRepBodies.item(0)
    check_bores(body, HOLES)
    if (not body.isSolid or body.lumps.count!=1 or
            abs(body.getPhysicalProperties(f.CalculationAccuracy.HighCalculationAccuracy).volume*1000
                -plate['volume_mm3'])>.1 or
            any(math.dist(a,b)>TOL for a,b in zip(layout.box(app,body),plate['bounds_mm']))):
        raise ValueError('Active plate differs from the verified V2.')
    cam = layout.find(parts['camera'],layout.CAMERA)
    lens = cam.component.bRepBodies.itemByName('Lens')
    caps = sorted([(face.pointOnFace.y,face.createForAssemblyContext(cam).centroid)
                   for face in lens.faces if c.Plane.cast(face.geometry) and abs(face.geometry.normal.y)>.99999],
                  key=lambda pair:pair[0])
    rear,front = [p for _,p in caps]
    angle = math.degrees(rear.vectorTo(front).angleTo(c.Vector3D.create(0,0,-1)))
    if angle>.01 or abs(front.z*10-recovery['lens']['lens_front_z_mm'])>TOL:
        raise ValueError('Live lens differs from registered pose.')
    report['lens']=dict(front_xyz_mm=[10*v for v in front.asArray()],angle_from_down_deg=angle)
    return body


def assess_trials(recovery,report):
    summaries={}
    for key,xy in [('reserved',[-15,45]),('equipment_envelope_midpoint',[97,45])]:
        location=recovery['placements'][key]
        trials=[t for t in location['trials'] if abs(t['adapter_height_mm']-47)<TOL]
        if math.dist(location['xy_mm'],xy)>TOL or sorted(t['beam_height_mm'] for t in trials)!=[30,50,80]:
            raise ValueError('Incomplete matching trial set: '+key)
        rows=summaries[key]=[]
        for t in trials:
            flange=[s for s in t['adapter_screws'] if s['interface']=='mounting_plate']
            if sorted(tuple(round(v,5) for v in s['xy_mm']) for s in flange)!=[(xy[0],xy[1]-25),(xy[0],xy[1]+25)]:
                raise ValueError('Adapter flange axes do not match trial.')
            # The V1 OD9 x 5 head/washer and OD10 x 30 driver envelopes above the
            # flange are retained exactly. The lower nut and socket are eliminated.
            upper={str(s['xy_mm']):{k:s['access'][k] for k in ('head+washer','driver above flange')} for s in flange}
            obstructions=[s['rod'] for s in t['sockets'] if s['side']=='lower' and s['assembled_axial_path_hits']]
            relevant_hits=bool(t['cross_equipment_hits'] or t['clamp_adapter_internal_hits'] or
                               any(any(paths.values()) for paths in upper.values()) or
                               any(s['pre_plate_axial_path_hits'] for s in t['sockets']) or
                               any(t['added_parts_in_service_corridors'].values()) or t['added_parts_ahead_of_lens'])
            rows.append(dict(beam_height_mm=t['beam_height_mm'],access_from_above=upper,
                former_nut_hits=[s['access']['nut+washer'] for s in flange],
                minimum_rod_or_nut_gap_mm=min(min(v['to_adapter_mm'],v['to_entire_plate_mm'])
                    for v in t['hardware_clearances_mm'] if v['name'].startswith(('Rod_','Nut_'))),
                blocked_lower_sockets_after_plate=obstructions,usb_stub_gap_mm=t['usb_stub_to_added_parts_mm'],
                usb_gap_bound_over_slot_travel_mm=t['usb_stub_lower_bound_for_camera_slot_travel_mm'],
                service_hits=t['added_parts_in_service_corridors'],lens_half_space_candidates=t['added_parts_ahead_of_lens'],
                clear_in_planned_sequence=not relevant_hits))
    report['stations_from_registered_recovery']=summaries
    report['balance']=dict(old_xy_mm=[-15,45],new_xy_mm=[97,45],distance_to_envelope_midpoint_mm=[112,0],
        actual_cg_and_moment='Unknown until equipment masses and centre of gravity are measured.')
    return all(t['clear_in_planned_sequence'] for t in summaries['equipment_envelope_midpoint'])


def build(app,r):
    recovery,refs=evidence(r)
    v2=helper('CreateMountingPlateV2')
    geom=helper('CreateCameraMountV3')
    layout=helper('CreateMountingLayoutV1')
    plate=refs['mounting-plate-v2']
    source=live_source(app,layout,recovery,plate,r)
    if math.dist(recovery['standalone_adapter']['size_mm'],[24,70,47])>TOL:
        raise ValueError('Unexpected adapter envelope')
    sources_clear=assess_trials(recovery,r)
    doc=app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
    doc.name='VISION SWINE Mounting Plate V3'
    design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent!=f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent=f.DesignIntentTypes.HybridDesignIntentType
    design.designType=f.DesignTypes.ParametricDesignType
    occ=design.rootComponent.occurrences.addNewComponent(c.Matrix3D.create())
    base=r['v2_geometry']={}
    v2.create_plate(design,occ,geom,layout,plate['bores'],base,[])
    comp=occ.component
    comp.name='Mounting_Plate_V3'
    body=comp.bRepBodies.item(0)
    body.name='Mounting Plate V3'
    # Both plates use the assembly XY axes and underside Z=0.
    manager=f.TemporaryBRepManager.get()
    for a,b in [(source,body),(body,source)]:
        # Intersect A with the complement of B inside enclosing stock.
        stock=c.OrientedBoundingBox3D.create(geom.point(81,44,3),layout.V(1,0,0),
                                              layout.V(0,1,0),24,12,2)
        difference=manager.createBox(stock)
        if (not manager.booleanOperation(difference,b,f.BooleanTypes.DifferenceBooleanType)
                or not manager.booleanOperation(difference,a,f.BooleanTypes.IntersectionBooleanType)
                or not difference.isValid or difference.lumps.count or difference.faces.count):
            raise ValueError('V3 blank differs from live V2 geometry.')
    params={'mp3X':'97 mm','mp3Y':'45 mm','mp3Pitch':'50 mm','mp3PilotD':'5.6 mm','mp3Pocket':'4.5 mm'}
    for name,expr in params.items():
        design.userParameters.add(name,geom.value(expr),'mm','Central M4 insert trial')
    top=geom.offset(comp,comp.xYConstructionPlane,'mp2T')
    xplane=geom.offset(comp,comp.yZConstructionPlane,'mp3X' if comp.yZConstructionPlane.geometry.normal.x>0 else '-mp3X')
    r['pockets']=[]
    for sign in ('-','+'):
        expr='mp3Y'+sign+'mp3Pitch/2'
        y=design.unitsManager.evaluateExpression(expr,'mm')*10
        yplane=geom.offset(comp,comp.xZConstructionPlane,expr if comp.xZConstructionPlane.geometry.normal.y>0 else '-('+expr+')')
        ref=geom.anchor(comp,top,xplane,yplane)
        sk=comp.sketches.add(top)
        centre=geom.linked_point(sk,ref)
        circle=sk.sketchCurves.sketchCircles.addByCenterRadius(centre,.28)
        sk.sketchDimensions.addDiameterDimension(circle,geom.point(102,y+5)).parameter.expression='mp3PilotD'
        before=geom.volume(comp)
        cut=geom.extrude(comp,sk,'mp3Pocket',f.ExtentDirections.NegativeExtentDirection,
                         f.FeatureOperations.CutFeatureOperation,'Blind M4 insert seat '+sign)
        wall,=[face for face in cut.faces if c.Cylinder.cast(face.geometry)]
        cyl=wall.geometry
        cx,cy=cyl.origin.x*10,cyl.origin.y*10
        zmin,zmax=wall.boundingBox.minPoint.z*10,wall.boundingBox.maxPoint.z*10
        d=cyl.radius*20
        removed=before-geom.volume(comp)
        if (math.dist([cx,cy,d,zmin,zmax],[97,y,5.6,1.5,6])>TOL or abs(cyl.axis.z)<.99999 or
                abs(removed-math.pi*(d/2)**2*(zmax-zmin))>.1):
            raise ValueError('Pilot cylinder or removed material differs from design.')
        flange=recovery['parameters_mm']['avFlange']
        # Trial flush M4 insert L4, M4x10 under-head length, 1 mm washer.
        penetration=10-flange-1
        tip=zmax-penetration
        if not (0<penetration<=4 and tip>zmin and zmax-4>zmin):
            raise ValueError('Insert or screw would bottom out.')
        r['pockets'].append(dict(xy_mm=[cx,cy],pilot_d_mm=d,z_range_mm=[zmin,zmax],
            floor_mm=zmin,insert_bottom_z_mm=zmax-4,screw_tip_z_mm=tip,
            nominal_engagement_mm=penetration,tip_to_pocket_floor_mm=tip-zmin,removed_mm3=removed))
    if not design.computeAll() or comp.bRepBodies.count!=1 or not body.isSolid or body.lumps.count!=1:
        raise ValueError('Incomplete V3 solid.')
    check_bores(body,HOLES)
    r.update(bounds_mm=layout.box(app,body),volume_mm3=geom.volume(comp),single_connected_solid=True,
        original_through_bores_xy_mm=HOLES,thickness_mm=6,
        adapter_xy_height_mm=[97,45,47],hardware_trial='Flush M4 insert length 4, pilot diameter 5.6 and depth 4.5; M4x10 with 1 mm washer. OD9 x 5 combined head/washer and OD10 x 30 driver envelopes.',
        changed_interference_basis='Only material removed from V2. Insert lies within original plate thickness. Screw shank OD4 fits OD4.5 flange; upper head and driver bounded by the recovered station envelopes. No hardware below Z=0.',
        reinforcement='6 mm fits the trial stack geometrically. If actual insert depth or pull-out capacity is insufficient, evaluate upward local reinforcement and revise the flange seat, screw length and clamp/socket clearances before use.',
        sequence=['Install flush inserts from above before fitting electronics.',
                  'Fasten adapter to lower clamp plate before setting beam. Tighten M8 nuts before attaching equipment plate.',
                  'Fit the plate beneath the flange. Install both M4x10 screws and washers from above. Remove plate before servicing obstructed lower M8 nuts.'],
        limitations=['Confirm insert supplier pilot, OD, seating, usable thread and PETG pull-out strength; screw and insert tolerances unmeasured.',
                     'Station and cable clearances refer to the verified geometry and USB stub, not real cable bend radius.'],
        status='cad_candidate_hardware_review_required' if sources_clear else 'access_review_required')
    app.activeViewport.fit()


def run(context):
    app=c.Application.get()
    r=dict(status='started',units='mm',approved_for_print=False,approved_for_installation=False)
    try:
        build(app,r)
    except Exception:
        r.update(status='incomplete',error=traceback.format_exc())
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=DOCS,delete=False) as out:
        json.dump(r,out,indent=2,allow_nan=False)
    os.replace(out.name,REPORT)
    app.userInterface.messageBox(r['status']+'\n'+str(REPORT))
