"""VISION SWINE prototype V1. API geometry uses cm; report and expressions use mm."""
import hashlib
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path

import adsk
import adsk.core as c
import adsk.fusion as f

PROJECT = Path(__file__).resolve().parents[2]
DOCS = PROJECT / 'Documentation'
PCB_NAME = '142-13449-1000-B01-SKU0000'
HOLES = [(4, 17), (90, 17), (4, 75), (90, 75)]
X = c.Vector3D.create
value = c.ValueInput.createByString


def point(x, y, z=0):
    return c.Point3D.create(x / 10, y / 10, z / 10)


def bounds(app, body, origin):
    box = app.measureManager.getOrientedBoundingBox(body, X(1, 0, 0), X(0, 1, 0))
    center = [10*a-b for a, b in zip(box.centerPoint.asArray(), origin)]
    half = [5*box.length, 5*box.width, 5*box.height]
    return ([a-b for a, b in zip(center, half)], [a+b for a, b in zip(center, half)])


def overlap(a, b):
    return all(min(a[1][i], b[1][i])-max(a[0][i], b[0][i]) > 0.00001 for i in range(3))


def name(body):
    return (body.assemblyContext.fullPathName if body.assemblyContext else 'root') + '/' + body.name


def interference(design, first, second):
    pair = c.ObjectCollection.create()
    pair.add(first)
    pair.add(second)
    query = design.createInterferenceInput(pair)
    query.areCoincidentFacesIncluded = False
    hits = design.analyzeInterference(query)
    if hits is None:
        raise RuntimeError('Fusion did not return interference results')
    return hits.count


def position(sk, vertex, x_expr, y_expr):
    g = vertex.geometry
    for orientation, expression in [
        (f.DimensionOrientations.HorizontalDimensionOrientation, x_expr),
        (f.DimensionOrientations.VerticalDimensionOrientation, y_expr)
    ]:
        dim = sk.sketchDimensions.addDistanceDimension(sk.originPoint, vertex, orientation,
                                                       c.Point3D.create(g.x+0.4, g.y+0.4, 0))
        dim.parameter.expression = expression


def extrude(comp, profile, expr, operation, title):
    features = comp.features.extrudeFeatures
    inp = features.createInput(profile, operation)
    extent = f.DistanceExtentDefinition.create(value(expr))
    if not inp.setOneSideExtent(extent, f.ExtentDirections.NegativeExtentDirection):
        raise RuntimeError('Cannot set extrusion extent')
    feature = features.add(inp)
    feature.name = title
    return feature


def create_plate(comp):
    planes = comp.constructionPlanes
    inp = planes.createInput()
    inp.setByOffset(comp.xYConstructionPlane, value('-vsPostH'))
    sk = comp.sketches.add(planes.add(inp))
    sk.name = 'Plate outline'
    lines = sk.sketchCurves.sketchLines.addTwoPointRectangle(point(-4,-4), point(104,83))
    # Bind opposite corners by their actual sketch coordinates, independent of edge order.
    corners = [line.startSketchPoint for line in lines] + [line.endSketchPoint for line in lines]
    position(sk, min(corners, key=lambda p:p.geometry.x+p.geometry.y), 'vsMargin', 'vsMargin')
    position(sk, max(corners, key=lambda p:p.geometry.x+p.geometry.y),
             'vsOuterX-vsMargin', 'vsOuterY-vsMargin')
    plate = extrude(comp, sk.profiles.item(0), 'vsFloor', f.FeatureOperations.NewBodyFeatureOperation,
                    'Base plate')
    body = plate.bodies.item(0)
    edges = c.ObjectCollection.create()
    for edge in body.edges:
        if abs(edge.startVertex.geometry.z-edge.endVertex.geometry.z) > 0.01:
            edges.add(edge)
    if edges.count != 4:
        raise ValueError('Expected four upright plate edges')
    fillets = comp.features.filletFeatures
    inp = fillets.createInput()
    inp.edgeSetInputs.addConstantRadiusEdgeSet(edges, value('vsCornerR'), False)
    fillets.add(inp).name = 'Outer corner radius'
    return body


def add_post(comp, index, x, y):
    sk = comp.sketches.add(comp.xYConstructionPlane)
    sk.name = f'Trial support {index} footprint'
    circles = sk.sketchCurves.sketchCircles
    outer = circles.addByCenterRadius(point(x,y), 0.275)
    inner = circles.addByCenterRadius(outer.centerSketchPoint, 0.1375)
    position(sk, outer.centerSketchPoint, f'vsHoleX{index}', f'vsHoleY{index}')
    for circle, expr in [(outer,'vsPostOD'), (inner,'vsBore')]:
        sk.sketchDimensions.addDiameterDimension(circle, point(x+4,y+4)).parameter.expression = expr
    ring = next(p for p in sk.profiles if p.profileLoops.count == 2)
    extrude(comp, ring, 'vsPostH', f.FeatureOperations.JoinFeatureOperation, f'Trial support {index}')


def inspect_site(app, pcb, bodies, origin, xy):
    """Use precise solid intersections in assembly space. Never operate on a design body as target."""
    mgr = f.TemporaryBRepManager.get()
    x, y = xy
    def cylinder(radius, low, high):
        a, b = [point(x+origin[0], y+origin[1], z+origin[2]) for z in (low, high)]
        return mgr.createCylinderOrCone(a, radius/10, b, radius/10)

    def subtract(target, tool):
        if not mgr.booleanOperation(target, tool, f.BooleanTypes.DifferenceBooleanType):
            raise RuntimeError('Temporary solid subtraction failed')

    def intersection_volume(probe, body):
        copy = mgr.copy(body)
        bb = bounds(app, copy, origin)
        expected = bounds(app, body, origin)
        if any(abs(a-b)>0.001 for side, ref in zip(bb,expected) for a,b in zip(side,ref)):
            raise ValueError('Transient copy frame mismatch: '+name(body))
        if not mgr.booleanOperation(probe, copy, f.BooleanTypes.IntersectionBooleanType):
            raise ValueError('Fusion could not evaluate solid intersection')
        return abs(probe.volume)*1000 if probe.faces.count else 0

    wall = False
    for face in pcb.faces:
        cyl = c.Cylinder.cast(face.geometry)
        if cyl and abs(cyl.axis.z)>0.999999 and abs(20*cyl.radius-2.75)<0.001:
            if math.hypot(10*cyl.origin.x-origin[0]-x,10*cyl.origin.y-origin[1]-y)<0.001:
                wall = True
                break
    if not wall:
        return dict(eligible=False, reason='No matching coaxial PCB cylinder')
    # Full-depth interior and nearly full contact annulus; 0.001 mm radial kernel allowance.
    void = intersection_volume(cylinder(1.374, -0.01, 1.58), pcb)
    seat = cylinder(2.75, 0, 0.05)
    subtract(seat, cylinder(1.376, -0.01, 0.06))
    expected = seat.volume*1000
    covered = intersection_volume(seat, pcb)
    collisions = []
    box = ([x-2.75,y-2.75,-5],[x+2.75,y+2.75,0])
    for body, bb in bodies:
        if body != pcb and overlap(box,bb):
            post = cylinder(2.75,-5,0)
            subtract(post,cylinder(1.375,-5.01,0.01))
            if intersection_volume(post,body)>1e-6:
                collisions.append(name(body))
    return dict(eligible=void<1e-6 and abs(covered-expected)<1e-5 and not collisions,
                bore_obstruction_mm3=void, bearing_expected_mm3=expected,
                bearing_observed_mm3=covered, post_interferences=collisions)


def build(app, report):
    inspection = json.loads((DOCS/'fusion-inspection.json').read_text(encoding='utf-8-sig'))
    ref = json.loads((DOCS/'step-inspection.json').read_text(encoding='utf-8-sig'))
    source = PROJECT/'Reference'/inspection['source']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != ref['sha256'] or inspection['units'] != 'mm':
        raise ValueError('STEP identity or units differ from inspection')
    frame = inspection['frame']
    if [frame[k] for k in 'xyz'] != [[1,0,0],[0,1,0],[0,0,1]]:
        raise ValueError('Prototype V1 expects the inspected assembly axes')
    origin = frame['origin_assembly_mm']
    report.update(step_sha256=digest, frame=frame, units='mm')
    manager = app.importManager
    doc = manager.importToNewDocument(manager.createSTEPImportOptions(str(source)))
    if not doc:
        raise RuntimeError('STEP import failed')
    doc.activate()
    adsk.doEvents()
    if app.activeDocument != doc:
        raise RuntimeError('Imported document not active')
    design = f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent != f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent = f.DesignIntentTypes.HybridDesignIntentType
    design.designType = f.DesignTypes.ParametricDesignType
    root = design.rootComponent
    solids = [b for b in root.bRepBodies if b.isSolid]
    for occ in root.allOccurrences:
        solids.extend(b for b in occ.bRepBodies if b.isSolid)
    boards = [b for b in solids if b.assemblyContext and
              b.assemblyContext.fullPathName == inspection['pcb_occurrence'] and
              PCB_NAME in name(b)]
    bodies = [(b,bounds(app,b,origin)) for b in solids]
    boards = [b for b,bb in bodies if b in boards and
              all(abs(u-v)<0.001 for s,k in zip(bb,('min','max'))
                  for u,v in zip(s,inspection['pcb'][k]))]
    if len(boards)!=1:
        raise ValueError('PCB identity or measured frame did not match the inspection')
    pcb = boards[0]
    overhang = []
    for body, bb in bodies:
        ext = [max(0,-bb[0][0]), max(0,bb[1][0]-100), max(0,-bb[0][1]), max(0,bb[1][1]-79)]
        if max(ext)>0.001:
            overhang.append(dict(body=name(body), bounds_mm=bb, overhang_minusX_plusX_minusY_plusY=ext))
    report.update(overhanging_bodies=overhang, identified_ports=inspection.get('ports',[]),
                  port_identity_complete=inspection.get('ports_complete',False))
    tr = c.Matrix3D.create()
    tr.translation = X(*(v/10 for v in origin))
    occurrence = root.occurrences.addNewComponent(tr)
    comp = occurrence.component
    comp.name = 'Jetson_Nano_Bottom_Case'
    params = design.userParameters
    expressions = dict(vsPcbX='100 mm',vsPcbY='79 mm',vsPcbT='1.57 mm',vsGap='1 mm',
                       vsWall='3 mm',vsFloor='3 mm',vsPostH='5 mm',vsCornerR='3 mm',
                       vsPostOD='5.5 mm',vsBore='2.75 mm',vsMargin='vsGap+vsWall',
                       vsInnerX='vsPcbX+2*vsGap',vsInnerY='vsPcbY+2*vsGap',
                       vsOuterX='vsInnerX+2*vsWall',vsOuterY='vsInnerY+2*vsWall')
    for index,(x,y) in enumerate(HOLES,1):
        expressions.update({f'vsHoleX{index}':f'{x} mm',f'vsHoleY{index}':f'{y} mm'})
    for key,expr in expressions.items():
        params.add(key,value(expr),'mm','Prototype V1')
    report['initial_parameter_expressions'] = expressions
    create_plate(comp)
    report['trial_sites'] = []
    for index,xy in enumerate(HOLES,1):
        entry = dict(center_xy=xy, post_created=False)
        report['trial_sites'].append(entry)
        try:
            entry.update(inspect_site(app,pcb,bodies,origin,xy))
        except Exception as exc:
            entry.update(eligible=False,verification_error=str(exc))
        if entry['eligible']:
            add_post(comp,index,*xy)
            entry['post_created'] = True
    case = occurrence.bRepBodies.item(0)
    bb = bounds(app,case,origin)
    report['case_bounds_mm'] = bb
    expected = [-4, -4, -8, 104, 83, 0 if any(site['post_created'] for site in report['trial_sites']) else -5]
    if any(abs(a-b)>0.001 for a,b in zip(bb[0]+bb[1], expected)):
        raise ValueError('Case extents differ from the inspection frame')
    collisions = report['case_board_interferences'] = []
    for body, envelope in bodies:
        if overlap(bb,envelope) and interference(design,case,body):
            collisions.append(name(body))
    report['status'] = 'review_required'
    app.activeViewport.fit()


def run(context):
    app = c.Application.get()
    path = DOCS/'bottom-case-v1-report.json'
    report = dict(status='started', timestamp=datetime.now().isoformat(), approved_for_print=False,
                  mount_function='experimental; mechanical purpose unconfirmed',
                  scope='As-built initial parameters only; review again after edits.')
    try:
        build(app,report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    finally:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(tmp,path)
    app.userInterface.messageBox(f"{report['status']}\n{path}")


