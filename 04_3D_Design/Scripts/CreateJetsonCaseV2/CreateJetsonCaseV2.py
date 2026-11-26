"""Jetson B01 open-front case. All report coordinates are PCB-relative mm."""
import hashlib
import importlib.util
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import adsk
import adsk.core as c
import adsk.fusion as f

PROJECT = Path(__file__).resolve().parents[2]
DOCS = PROJECT / 'Documentation'


def read_report(filename):
    return json.loads((DOCS / filename).read_text(encoding='utf-8-sig'))


def load_base_tools():
    path = PROJECT / 'Scripts/CreateJetsonBottomCase/CreateJetsonBottomCase.py'
    spec = importlib.util.spec_from_file_location('vision_swine_base_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_walls(base, comp, plate, width, depth, gap, margin):
    # Extending the existing rounded face keeps all outside radii associative.
    face = max(plate.faces, key=lambda face: face.pointOnFace.z)
    extrudes = comp.features.extrudeFeatures
    inp = extrudes.createInput(face, f.FeatureOperations.JoinFeatureOperation)
    distance = f.DistanceExtentDefinition.create(base.value('vsWallH'))
    if not inp.setOneSideExtent(distance, f.ExtentDirections.PositiveExtentDirection):
        raise RuntimeError('Cannot extend top face')
    extrudes.add(inp).name = 'Wall exterior from rounded plate'
    planes = comp.constructionPlanes
    offset = planes.createInput()
    offset.setByOffset(comp.xYConstructionPlane, base.value('vsWallH-vsPostH'))
    sk = comp.sketches.add(planes.add(offset))
    sk.name = 'Open front cavity'
    lines = sk.sketchCurves.sketchLines.addTwoPointRectangle(
        base.point(-gap, -2*margin), base.point(width+gap, depth+gap))
    corners = [p for line in lines for p in (line.startSketchPoint, line.endSketchPoint)]
    base.position(sk, min(corners,key=lambda p:p.geometry.x+p.geometry.y),'vsGap','2*vsMargin')
    base.position(sk, max(corners,key=lambda p:p.geometry.x+p.geometry.y),'vsPcbX+vsGap','vsPcbY+vsGap')
    base.extrude(comp, sk.profiles.item(0), 'vsWallH', f.FeatureOperations.CutFeatureOperation,
                 'Open minus Y and hollow three walls')


def swept_box(bb, axis, end, sign):
    lo, hi = [list(side) for side in bb]
    if sign < 0:
        lo[axis] = min(end,lo[axis])
    else:
        hi[axis] = max(end,hi[axis])
    return [lo,hi]


def micro_sd(base, app, bodies, case_bb, obstacles):
    """Use an actual card body when present; never substitute a socket or the PCB."""
    candidates = [(body,bb) for body,bb in bodies if any(
        tag in base.name(body).lower() for tag in ('microsd','micro_sd','micro-sd','sd_card'))]
    result = dict(status='unverified', labelled_candidates=[base.name(b) for b,_ in candidates])
    ui = app.userInterface
    if ui.messageBox('Is a separate microSD CARD body visible and its straight removal direction known? '
                     'Yes: select the card to check case clearance. No: record unverified.',
                     'MicroSD access', c.MessageBoxButtonTypes.YesNoButtonType) != c.DialogResults.DialogYes:
        result['reason'] = 'Card body or ejection direction not confirmed'
        return result
    try:
        selection = ui.selectEntity('Select the removable card solid itself','SolidBodies')
        selected = selection.entity if selection else None
        matches = [(b,bb) for b,bb in bodies if b == selected]
        if len(matches)!=1:
            result['reason'] = 'No unique imported card solid selected'
            return result
        body,bb = matches[0]
        direction,cancelled = ui.inputBox('PCB-frame ejection axis: +X, -X, +Y or -Y','MicroSD direction')
        direction = direction.strip().upper()
        if cancelled or direction not in ('+X','-X','+Y','-Y'):
            result['reason'] = 'No verified axis supplied'
            return result
        axis, sign = 'XY'.index(direction[1]), (1 if direction[0]=='+' else -1)
        end = case_bb[1 if sign>0 else 0][axis]+sign
        corridor = swept_box(bb,axis,end,sign)
        possible = [label for label,box in obstacles if base.overlap(box,corridor)]
        result.update(status='potential_case_obstruction' if possible else 'card_envelope_clear_of_case',
                      body=base.name(body), direction=direction, swept_bounds=corridor,
                      possible_obstacles=possible, finger_access='unverified')
    except RuntimeError as exc:
        result['reason'] = str(exc)
    return result


def build(base, app, report):
    insp = read_report('fusion-inspection.json')
    old = read_report('bottom-case-v1-report.json')
    source = PROJECT/'Reference'/insp['source']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if (digest != old['step_sha256'] or old['frame'] != insp['frame'] or
            insp['units'] != 'mm' or old['units'] != 'mm'):
        raise ValueError('Inspection and V1 references disagree')
    frame = insp['frame']
    if [frame[k] for k in 'xyz'] != [[1,0,0],[0,1,0],[0,0,1]]:
        raise ValueError('Unexpected assembly axes')
    sites = old['trial_sites']
    if ([tuple(s['center_xy']) for s in sites] != base.HOLES or
            not all(s['eligible'] and s['post_created'] for s in sites)):
        raise ValueError('Expected the four V1 trial supports')
    origin = frame['origin_assembly_mm']
    report.update(frame=frame, step_sha256=digest)
    mgr = app.importManager
    doc = mgr.importToNewDocument(mgr.createSTEPImportOptions(str(source)))
    if not doc:
        raise RuntimeError('STEP import failed')
    doc.activate()
    adsk.doEvents()
    if app.activeDocument != doc:
        raise RuntimeError('Imported document is not active')
    design = f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent != f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent = f.DesignIntentTypes.HybridDesignIntentType
    design.designType = f.DesignTypes.ParametricDesignType
    root = design.rootComponent
    solids = [b for b in root.bRepBodies if b.isSolid]
    for occ in root.allOccurrences:
        solids.extend(b for b in occ.bRepBodies if b.isSolid)
    bodies = [(b,base.bounds(app,b,origin)) for b in solids]
    def near(a,b):
        return len(a)==len(b) and all(abs(x-y)<0.001 for x,y in zip(a,b))
    if not any(b.assemblyContext and b.assemblyContext.fullPathName==insp['pcb_occurrence']
               and near(bb[0],insp['pcb']['min']) and near(bb[1],insp['pcb']['max']) for b,bb in bodies):
        raise ValueError('PCB frame mismatch')
    transform = c.Matrix3D.create()
    transform.translation = c.Vector3D.create(*(v/10 for v in origin))
    occ = root.occurrences.addNewComponent(transform)
    comp = occ.component
    comp.name = 'Jetson_Nano_Case_V2'
    params = design.userParameters
    for key,expr in dict(old['initial_parameter_expressions'], vsWallH='20 mm').items():
        params.add(key,base.value(expr),'mm','VISION SWINE V2')
    def mm(key):
        return params.itemByName(key).value*10
    width,depth,gap,margin,floor_z,height = (mm('vsPcbX'),mm('vsPcbY'),mm('vsGap'),
                                              mm('vsMargin'),-mm('vsPostH'),mm('vsWallH'))
    plate = base.create_plate(comp)
    add_walls(base,comp,plate,width,depth,gap,margin)
    for index,(x,y) in enumerate(base.HOLES,1):
        base.add_post(comp,index,x,y)
    design.computeAll()
    if occ.bRepBodies.count != 1:
        raise RuntimeError('Expected exactly one case body')
    case = occ.bRepBodies.item(0)
    bb = base.bounds(app,case,origin)
    top = floor_z+height
    solid_ok = case.isValid and case.isSolid and case.lumps.count==1
    report.update(case_bounds_mm=bb, single_connected_solid=solid_ok)
    if not solid_ok or not near(bb[0],old['case_bounds_mm'][0]) or not near(bb[1],[104,83,top]):
        raise ValueError('Case topology or extents failed V2 verification')
    mid_z = floor_z + height/2
    def inside(x,y,z):
        test = base.point(origin[0]+x,origin[1]+y,origin[2]+z)
        return case.pointContainment(test) == f.PointContainment.PointInsidePointContainment
    wall_mid = (margin+gap)/2
    samples = {'minus_x':inside(-wall_mid,depth/2,mid_z),
               'plus_x':inside(width+wall_mid,depth/2,mid_z),
               'plus_y':inside(width/2,depth+wall_mid,mid_z),
               'open_minus_y':not inside(width/2,-wall_mid,mid_z),
               'clear_interior':not inside(width/2,depth/2,mid_z)}
    for i,(x,y) in enumerate(base.HOLES,1):
        samples[f'blind_support_{i}'] = (not inside(x,y,floor_z/2)
            and inside(x+(mm('vsPostOD')+mm('vsBore'))/4,y,floor_z/2)
            and inside(x,y,floor_z-mm('vsFloor')/2))
    report['geometry_samples'] = samples
    if not all(samples.values()):
        raise ValueError('Wall, opening or blind support geometry check failed')
    lo,hi = bb
    def box(x0,y0,z0,x1,y1,z1):
        return [[x0,y0,z0],[x1,y1,z1]]
    # Boxes enclose all case material; rounded corners and blind bores make tests conservative.
    obstacles = [('floor',box(*lo,hi[0],hi[1],floor_z)),
                 ('-X wall',box(lo[0],lo[1],floor_z,-gap,hi[1],top)),
                 ('+X wall',box(width+gap,lo[1],floor_z,hi[0],hi[1],top)),
                 ('+Y wall',box(lo[0],depth+gap,floor_z,hi[0],hi[1],top))]
    radius = mm('vsPostOD')/2
    for index,(x,y) in enumerate(base.HOLES,1):
        obstacles.append((f'post {index}',box(x-radius,y-radius,floor_z,x+radius,y+radius,0)))
    collisions = []
    for body, envelope in bodies:
        if any(base.overlap(envelope,o) for _,o in obstacles) and base.interference(design,case,body):
            collisions.append(base.name(body))
    report['case_board_interferences'] = collisions
    lift = top-min(envelope[0][2] for _,envelope in bodies)+1
    possible = []
    for body,envelope in bodies:
        swept = swept_box(envelope,2,envelope[1][2]+lift,1)
        for label,o in obstacles:
            if not base.overlap(swept,o):
                continue
            # Actual static fit is known; holes in posts / irregular solids can make AABBs inconclusive.
            possible.append(dict(body=base.name(body), obstacle=label))
    report['top_install_remove'] = dict(method='continuous vertical swept body bounding boxes; bare board, fasteners removed',
        travel_mm=lift, status='inconclusive' if possible or collisions else 'clear', potential_obstructions=possible)
    front = []
    for body,envelope in bodies:
        if envelope[0][1]>=-0.001:
            continue
        corridor = swept_box(envelope,1,lo[1]-1,-1)
        hits = [label for label,o in obstacles if base.overlap(corridor,o)]
        front.append(dict(body=base.name(body), housing_sweep=corridor, possible_case_obstacles=hits))
    report['minus_y_access'] = dict(housing_corridors=front,
        physical_plug_and_finger_clearance='unverified; plug dimensions, grip space and cable bend radii not supplied')
    report['micro_sd_access'] = micro_sd(base,app,bodies,bb,obstacles)
    report['status'] = 'interference_detected' if collisions else 'review_required'
    app.activeViewport.fit()


def run(context):
    app = c.Application.get()
    path = DOCS/'jetson-case-v2-report.json'
    report = dict(status='started',timestamp=datetime.now().isoformat(),units='mm',
                  approved_for_print=False, scope='Initial as-built geometry only')
    try:
        build(load_base_tools(),app,report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    finally:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(tmp,path)
    app.userInterface.messageBox(f"{report['status']}\n{path}")

