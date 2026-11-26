"""Build in a new Fusion document. All report coordinates are assembly millimetres."""
import importlib.util
import json
import math
import os
import tempfile
import time
import traceback
from pathlib import Path

import adsk.core as c
import adsk.fusion as f
import adsk

SCRIPTS = Path(__file__).resolve().parents[1]
REPORT = SCRIPTS.parent / 'Documentation' / 'mounting-plate-v1-report.json'
TOL = 0.01


def helpers(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / name / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layout_parts(parent, layout):
    children = parent.childOccurrences if f.Occurrence.cast(parent) else parent.occurrences
    if children.count != 2:
        raise ValueError('Select the original two-assembly Mounting Layout V1.')
    result = {}
    for child in children:
        labels = {o.component.name for o in child.childOccurrences}
        if {layout.CASE, layout.PANEL, layout.LID} <= labels:
            result['case'] = child
        if {layout.BRACKET, layout.CAMERA, layout.HEAD} <= labels:
            result['camera'] = child
    if len(result) != 2:
        raise ValueError('Missing full V5 or V3 assembly in the selected layout.')
    return result


def create_plate(design, geom, layout, points, report):
    occ = design.rootComponent.occurrences.addNewComponent(c.Matrix3D.create())
    comp = occ.component
    comp.name = 'Mounting_Plate_V1'
    params = dict(mpX='-30 mm', mpY='-8 mm', mpMainW='170 mm', mpMainH='106 mm',
                  mpTongueY='1 mm', mpReach='47 mm', mpTongueW='36 mm', mpT='6 mm',
                  mpOuterR='3 mm', mpRootR='8 mm', mpLowerR='0.75 mm', mpHoleD='4.5 mm')
    for i, station in enumerate(points, 1):
        for axis, n in zip('XY', station['center_xyz_mm']):
            params[f'mpHole{i}{axis}'] = f'{n:.6f} mm'
    for key, expr in params.items():
        design.userParameters.add(key, geom.value(expr), 'mm', 'Mounting Plate V1')

    def mm(expr):
        return design.unitsManager.evaluateExpression(expr, 'mm') * 10

    xy = comp.xYConstructionPlane
    planes = {}

    def anchor(x, y):
        for axis, expr, base in [('x', x, comp.yZConstructionPlane), ('y', y, comp.xZConstructionPlane)]:
            if (axis, expr) not in planes:
                signed = expr if getattr(base.geometry.normal, axis) > 0 else '-(' + expr + ')'
                planes[axis, expr] = geom.offset(comp, base, signed)
        return geom.anchor(comp, xy, planes['x', x], planes['y', y])

    right = 'mpX+mpMainW'
    tip = right + '+mpReach'
    upper = 'mpTongueY+mpTongueW'
    top = 'mpY+mpMainH'
    corners = [('mpX', 'mpY'), (right, 'mpY'), (right, 'mpTongueY'), (tip, 'mpTongueY'),
               (tip, upper), (right, upper), (right, top), ('mpX', top)]
    radii = ['mpOuterR', 'mpLowerR', 'mpRootR', 'mpOuterR', 'mpOuterR', 'mpRootR',
             'mpOuterR', 'mpOuterR']
    coords = [(mm(x), mm(y)) for x, y in corners]
    for i, p in enumerate(coords):
        if math.dist(p, coords[i-1]) <= mm(radii[i]) + mm(radii[i-1]):
            raise ValueError('Adjacent corner radii do not fit on edge ' + str(i))
    refs = [anchor(x, y) for x, y in corners]
    sketch = comp.sketches.add(xy)
    sketch.name = 'Main plate and camera tongue'
    pts = [geom.linked_point(sketch, ref) for ref in refs]
    for i, start in enumerate(pts):
        sketch.sketchCurves.sketchLines.addByTwoPoints(start, pts[(i+1) % len(pts)])
    pos = f.ExtentDirections.PositiveExtentDirection
    geom.extrude(comp, sketch, 'mpT', pos, f.FeatureOperations.NewBodyFeatureOperation, 'Plate blank')
    body = comp.bRepBodies.item(0)
    body.name = 'Mounting Plate V1'
    groups = {key: c.ObjectCollection.create() for key in set(radii)}
    for (x, y), key in zip(coords, radii):
        edges = [e for e in body.edges if all(
            math.dist([v.geometry.x*10, v.geometry.y*10], [x, y]) < TOL
            for v in (e.startVertex, e.endVertex))]
        if len(edges) != 1:
            raise ValueError('Missing unique vertical edge at ' + str((x, y)))
        groups[key].add(edges[0])
    fillets = comp.features.filletFeatures
    inp = fillets.createInput()
    for key, edges in groups.items():
        inp.edgeSetInputs.addConstantRadiusEdgeSet(edges, geom.value(key), False)
    fillets.add(inp).name = 'Outside and tongue root radii'
    report['corner_radii_mm'] = [dict(xy=p, radius=mm(r)) for p, r in zip(coords, radii)]
    report['bores'] = []
    for i, station in enumerate(points, 1):
        x, y, _ = station['center_xyz_mm']
        ref = anchor(f'mpHole{i}X', f'mpHole{i}Y')
        sk = comp.sketches.add(xy)
        circle = sk.sketchCurves.sketchCircles.addByCenterRadius(geom.linked_point(sk, ref), mm('mpHoleD')/20)
        sk.sketchDimensions.addDiameterDimension(circle, geom.point(x+4, y+4)).parameter.expression = 'mpHoleD'
        before = geom.volume(comp)
        feat = geom.extrude(comp, sk, 'mpT', pos, f.FeatureOperations.CutFeatureOperation, f'M4 clearance {i}')
        walls = [face for face in feat.faces if c.Cylinder.cast(face.geometry)]
        if len(walls) != 1:
            raise ValueError('Missing unique cylindrical cut wall')
        wall = walls[0]
        cyl = wall.geometry
        cx, cy = cyl.origin.x*10, cyl.origin.y*10
        diameter = cyl.radius*20
        bb = wall.boundingBox
        removed = before - geom.volume(comp)
        ok = (abs(cyl.axis.z) > .99999 and math.dist((x,y),(cx,cy)) < TOL and
              abs(diameter-mm('mpHoleD')) < TOL and abs(bb.minPoint.z*10) < TOL and
              abs(bb.maxPoint.z*10-mm('mpT')) < TOL and
              abs(removed-math.pi*(diameter/2)**2*mm('mpT')) < .1)
        report['bores'].append(dict(center_xyz_mm=[cx,cy,0], diameter_mm=diameter,
                                    removed_mm3=removed, through_and_aligned=ok))
        if not ok:
            raise ValueError(f'Bore {i} failed measurement')
    design.computeAll()
    if comp.bRepBodies.count != 1 or not body.isSolid or body.lumps.count != 1:
        raise ValueError('Plate is not a single connected solid')
    world = occ.bRepBodies.item(0)
    bounds = layout.box(c.Application.get(), world)
    if any(math.dist(actual, expected) > TOL for actual, expected in zip(
            bounds, [[mm('mpX'), mm('mpY'), 0], [mm(tip), mm(top), mm('mpT')]])):
        raise ValueError('Plate bounds differ from design parameters')
    report.update(single_connected_solid=True, bounds_mm=bounds, volume_mm3=geom.volume(comp))
    manager = f.TemporaryBRepManager.get()
    sections = []
    for x in (mm(right)+.01, mm(right)+mm('mpRootR')+.01):
        wire = manager.planeIntersection(body, c.Plane.create(geom.point(x,0), layout.V(1,0,0)))
        face_body = manager.createFaceFromPlanarWires([wire])
        sections.append(dict(x_mm=x, area_mm2=sum(face.area for face in face_body.faces)*100))
    report['tongue_sections'] = sections
    return world


def hardware_checks(geom, layout, bodies, points, thickness, lower_seats):
    # Head above plate, nut on the far side of the ear/bracket. Positive-volume
    # intersections only. Sleeves bound a socket or driver on a straight approach.
    manager = f.TemporaryBRepManager.get()
    checks = []
    for i, station in enumerate(points):
        x, y, _ = station['center_xyz_mm']
        seat = lower_seats[station['part']]
        for label, od, start, end in [
                ('upper washer', 9, thickness, thickness+1),
                ('upper head', 8, thickness+1, thickness+5),
                ('upper driver', 10, thickness+5, thickness+35),
                ('lower washer', 9, seat-1, seat),
                ('lower nut envelope', 8.2, seat-4.2, seat-1),
                ('lower socket approach', 12, seat-34.2, seat-1)]:
            rad = od/2
            bb = [[x-rad,y-rad,start],[x+rad,y+rad,end]]
            env = manager.createCylinderOrCone(geom.point(x,y,start), rad/10,
                                               geom.point(x,y,end), rad/10)
            hits = []
            for body, bounds in bodies:
                if layout.overlaps(bb, bounds):
                    cut = manager.copy(env)
                    if not manager.booleanOperation(cut, body, f.BooleanTypes.IntersectionBooleanType):
                        raise ValueError('Cannot evaluate hardware space at hole ' + str(i+1))
                    if cut.lumps.count and cut.volume*1000 > .001:
                        hits.append(body.name)
            nearest = min(math.dist((x,y), p['center_xyz_mm'][:2]) for j,p in enumerate(points) if j != i)
            checks.append(dict(hole=i+1, envelope=label, od_mm=od, z_mm=[start,end],
                               interfering_bodies=hits,
                               lateral_margin_to_adjacent_washer_projection_mm=nearest-rad-4.5))
    return checks


def inspect_plate(app, design, layout, geom, parts, plate, live, v5, v3, report):
    pb = report['bounds_mm']
    thickness = pb[1][2]-pb[0][2]
    report['size_mm'] = [b-a for a,b in zip(*pb)]
    bodies = [(b,layout.box(app,b)) for part in parts.values() for b in layout.solids(part)]
    hits = []
    for b, bb in bodies:
        if layout.overlaps(bb,pb) and layout.interference(design,plate,b):
            hits.append(b.name)
    report['interference'] = dict(plate_against_layout=hits, solids_screened=len(bodies),
        case_against_camera=live['cross_assembly_interferences'],
        scope='Existing internal contacts in V5 and V3 remain covered by their source inspections.')
    report['mating_faces'] = live['mounting_faces']
    report['lens'] = dict(**live['view_check'], plate_ahead_of_lens=pb[0][2]<live['view_check']['lens_front_z_mm'])
    report['service_corridors'] = [
        dict(region=s['region'], bounds_mm=s['bounds'], camera_candidates=s['camera_candidates'],
             plate_bounds_overlap=layout.overlaps(pb,s['bounds'])) for s in live['service_access']['corridors']]
    report['ventilation'] = dict(lid_faces_down=True, plate_above_case=pb[0][2]>=live['case_bounds'][1][2]-TOL,
        side_intake_gap_mm=live['measured_gap_mm'], harness_and_thermal_tests='pending')
    camera = layout.find(parts['camera'], layout.CAMERA)
    usb = camera.bRepBodies.itemByName('USB cable')
    if not usb:
        raise ValueError('USB reference body unavailable')
    distance = app.measureManager.measureMinimumDistance(usb, plate).value*10
    points = live['mounting_points']
    slots = points[4:]
    # A nominal 4 mm shank can move slightly farther than the end-arc centres.
    travel = min((s['slot_length_mm']-4)/2 for s in slots)
    report['usb'] = dict(model_gap_mm=distance, lower_bound_over_slot_travel_mm=max(0,distance-travel),
                         real_plug_bend_radius_strain_relief_and_power_harness='unverified')
    bracket = layout.find(parts['camera'], layout.BRACKET)
    faces = [face for b in bracket.component.bRepBodies for face in b.faces
             if c.Plane.cast(face.geometry) and abs(face.geometry.normal.y)>.99999
             and abs(face.pointOnFace.y*10)<TOL]
    seat_face = max(faces,key=lambda face:face.area).createForAssemblyContext(bracket)
    seat_box = layout.box(app,seat_face)
    tongue = live['plate_proposal']['camera_tongue_xy']
    y0,y1 = tongue[0][1],tongue[1][1]
    support_y = [seat_box[0][1]-travel,seat_box[1][1]+travel]
    sweep_slots = [min(s['center_xyz_mm'][1]-s['slot_length_mm']/2 for s in slots)-travel,
                   max(s['center_xyz_mm'][1]+s['slot_length_mm']/2 for s in slots)+travel]
    report['camera_support'] = dict(mating_face_bounds=seat_box, adjustment_each_way_mm=travel,
        swept_face_y_mm=support_y, swept_slot_y_mm=sweep_slots,
        full_slot_margin_y_mm=[sweep_slots[0]-y0,y1-sweep_slots[1]],
        full_face_margin_y_mm=[support_y[0]-y0,y1-support_y[1]],
        fixed_washer_edge_margin_y_mm=min(min(s['center_xyz_mm'][1]-y0,y1-s['center_xyz_mm'][1])
                                          for s in slots)-4.5,
        proposed_if_full_face_backing_required=dict(y_limits_mm=support_y,
            minimum_tip_x_with_R3_mm=seat_box[1][0]+3,
            note='Alternative only. Rework the lower transition and recheck cable, corner radii and service sweep.'))
    ear_t = v5['mounting_plate_interface']['material_sections_mm']['root_thickness']
    lower = {'case':-ear_t, 'camera bracket':-v3['measured_thickness_mm']['vertical']}
    report['hardware'] = dict(assumptions='Provisional axial envelopes in mm, head above plate, washer on each side; nominal bracket position. Verify actual nut/socket, screw length and tightening sequence.',
        envelopes=hardware_checks(geom,layout,bodies+[(plate,pb)],points,thickness,lower))
    clamp = live['plate_proposal']['clamp_candidate_center_xy']
    report['beam_clamp'] = dict(reserved_xy_mm=live['plate_proposal']['clamp_zone_xy'],
        candidate_xy_mm=clamp, drilled=False,
        horizontal_offset_to_mount_pattern_centroids_mm={
            group:math.dist(clamp,[sum(p['center_xyz_mm'][k] for p in stations)/len(stations) for k in (0,1)])
            for group,stations in [('case',points[:4]),('camera',slots)]},
        load_evaluation='Clamp is to the left of both mounting patterns. Weigh the complete hardware and locate its horizontal CG. For vertical gravity, torque magnitude in N m = mass_kg * 9.80665 * horizontal_CG_offset_mm / 1000. Mount-pattern centres are geometric references only.')


def select_layout(app, layout, report):
    active = app.activeDocument
    states = report['open_documents'] = []
    report['active_document'] = active.name if active else None
    product = app.activeProduct
    report['active_product_type'] = product.objectType if product else None
    candidates = {}
    for number, doc in enumerate(app.documents, 1):
        state = dict(number=number, name=doc.name, active=doc.isActive,
                     isSaved=doc.isSaved, isModified=doc.isModified,
                     hasDataFile=None, isComplete=None, version=None)
        states.append(state)
        try:
            # Untitled documents may throw on dataFile instead of returning None.
            try:
                file = doc.dataFile
                state['hasDataFile'] = bool(file)
                if file:
                    state.update(version=file.versionNumber, isComplete=file.isComplete)
            except RuntimeError as exc:
                state['dataFile_error'] = str(exc)
                file = None
            design = f.Design.cast(doc.products.itemByProductType('DesignProductType'))
            if not design:
                raise ValueError('No Design product')
            layout_parts(design.rootComponent, layout)
            if not file:
                raise ValueError('Saved DataFile unavailable')
            candidates[number] = (doc, design, file, state)
            state['result'] = 'matching assembly'
        except (ValueError, RuntimeError) as exc:
            state['result'] = str(exc)

    lines = [f"{s['number']}. {s['name']} | saved={s['isSaved']}, modified={s['isModified']}, "
             f"DataFile={s['hasDataFile']}, v{s['version']}, complete={s['isComplete']} | {s['result']}"
             for s in states]
    if not candidates:
        raise ValueError('Open the saved VISION_SWINE_Mounting_Layout_V1. Open documents: '
                         + ' / '.join(lines))
    number = next(iter(candidates))
    if len(candidates) > 1 or candidates[number][3]['isModified']:
        prompt = ('Select the layout number to import its SAVED version. '
                  'Unsaved changes are excluded.\n' + '\n'.join(lines))
        while True:
            answer, cancelled = app.userInterface.inputBox(prompt, 'Mounting Layout V1', str(number))
            if cancelled:
                return None
            if answer.strip() in {str(n) for n in candidates}:
                number = int(answer)
                break
    doc, design, file, state = candidates[number]
    source = report['layout_source'] = dict(name=doc.name, id=file.id, version=file.versionNumber,
        isModified=state['isModified'], initial_isComplete=state['isComplete'])
    # Give an in-flight save time to finish, retaining the selected version.
    deadline = time.monotonic() + 30
    while not file.isComplete and time.monotonic() < deadline:
        adsk.doEvents()
        time.sleep(.2)
        file.refresh()
    source['isComplete'] = file.isComplete
    if not source['isComplete']:
        raise ValueError(f"{doc.name} v{source['version']}: DataFile.isComplete is still False "
                         "after 30 seconds. Check Fusion Job Status and rerun after processing.")
    return file, design


def verify_layout(parent, layout, baseline):
    roots = layout_parts(parent, layout)
    for key, node in roots.items():
        recorded = baseline['sources'][key]
        if layout.inventory(node) != recorded['inserted_inventory'] or any(
                abs(a-b) > 1e-6 for a,b in zip(node.transform2.asArray(),recorded['transform'])):
            raise ValueError('Layout inventory or placement differs from the report: ' + key)
    return roots


def build(app, report):
    layout = helpers('CreateMountingLayoutV1')
    geom = helpers('CreateCameraMountV3')
    baseline = layout.read('mounting-layout-v1-report.json')
    v5 = layout.read('jetson-case-v5-report.json')
    v3 = layout.read('camera-mount-v3-report.json')
    selected = select_layout(app,layout,report)
    if selected is None:
        report['status'] = 'cancelled'
        return
    file, src = selected
    # A modified tab may differ from the selected saved revision. Its component
    # signature was checked in select_layout; verify the saved geometry on insert.
    if not report['layout_source']['isModified']:
        verify_layout(src.rootComponent,layout,baseline)
    doc = app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
    doc.name = 'VISION SWINE Mounting Plate V1'
    design = f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent != f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent = f.DesignIntentTypes.HybridDesignIntentType
    if design.designType != f.DesignTypes.ParametricDesignType:
        design.designType = f.DesignTypes.ParametricDesignType
    try:
        copied = design.rootComponent.occurrences.addByInsert(file,c.Matrix3D.create(),False)
        if not copied or copied.bRepBodies.count:
            raise ValueError('Expected a saved layout containing only the two source assemblies')
        parts = verify_layout(copied,layout,baseline)
    except Exception:
        doc.close(False)
        raise
    live = {}
    layout.inspect(app,design,parts,v5,v3,baseline['measured_gap_mm'],live)
    points = live['mounting_points']
    expected = [(6,38,0),(6,28,0),(126,38,0),(126,28,0),(172,9,0),(172,29,0)]
    if len(points)!=6 or len(baseline['mounting_points'])!=6 or any(
            math.dist(p['center_xyz_mm'], q['center_xyz_mm'])>TOL or math.dist(p['center_xyz_mm'], e)>TOL
            for p,q,e in zip(points,baseline['mounting_points'],expected)):
        raise ValueError('Live mounting points differ from the requested layout')
    plate = create_plate(design,geom,layout,points,report)
    inspect_plate(app,design,layout,geom,parts,plate,live,v5,v3,report)
    app.activeViewport.fit()
    report['status'] = 'review_required'


def run(context):
    app = c.Application.get()
    report = dict(status='checking_layout', units='mm', approved_for_installation=False,
                  strength_tested=False, scope='As built on this run, before any later parameter edits.')
    try:
        build(app,report)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=REPORT.parent,
                                     delete=False, suffix='.tmp') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    os.replace(stream.name,REPORT)
    message = report.get('error','').splitlines()
    app.userInterface.messageBox('\n'.join([report['status'], *message[-1:], str(REPORT)]))
