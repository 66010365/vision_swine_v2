"""Build in the active Mounting Plate assembly. All report coordinates are assembly millimetres."""
import importlib.util
import json
import math
import os
import tempfile
import traceback
from pathlib import Path

import adsk.core as c
import adsk.fusion as f

SCRIPTS = Path(__file__).resolve().parents[1]
REPORT = SCRIPTS.parent / 'Documentation' / 'mounting-plate-v2-report.json'
TOL = 0.01


def helpers(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / name / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_plate(design, occ, geom, layout, points, report, created_params):
    comp = occ.component
    comp.name = 'Mounting_Plate_V2'
    params = dict(mp2X='-30 mm', mp2Y='-10 mm', mp2MainW='170 mm', mp2MainH='108 mm',
                  mp2Reach='52 mm', mp2TongueW='62 mm', mp2T='6 mm',
                  mp2OuterR='3 mm', mp2RootR='8 mm', mp2HoleD='4.5 mm')
    for i, station in enumerate(points, 1):
        for axis, n in zip('XY', station['center_xyz_mm']):
            params[f'mp2Hole{i}{axis}'] = f'{n:.6f} mm'
    for key, expr in params.items():
        created_params.append(design.userParameters.add(key, geom.value(expr), 'mm', 'Mounting Plate V2'))

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

    right = 'mp2X+mp2MainW'
    tip = right + '+mp2Reach'
    upper = 'mp2Y+mp2TongueW'
    top = 'mp2Y+mp2MainH'
    corners = [('mp2X', 'mp2Y'), (tip, 'mp2Y'),
               (tip, upper), (right, upper), (right, top), ('mp2X', top)]
    radii = ['mp2OuterR', 'mp2OuterR', 'mp2OuterR', 'mp2RootR',
             'mp2OuterR', 'mp2OuterR']
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
    geom.extrude(comp, sketch, 'mp2T', pos, f.FeatureOperations.NewBodyFeatureOperation, 'Plate blank')
    body = comp.bRepBodies.item(0)
    body.name = 'Mounting Plate V2'
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
        ref = anchor(f'mp2Hole{i}X', f'mp2Hole{i}Y')
        sk = comp.sketches.add(xy)
        circle = sk.sketchCurves.sketchCircles.addByCenterRadius(geom.linked_point(sk, ref), mm('mp2HoleD')/20)
        sk.sketchDimensions.addDiameterDimension(circle, geom.point(x+4, y+4)).parameter.expression = 'mp2HoleD'
        before = geom.volume(comp)
        feat = geom.extrude(comp, sk, 'mp2T', pos, f.FeatureOperations.CutFeatureOperation, f'M4 clearance {i}')
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
              abs(diameter-mm('mp2HoleD')) < TOL and abs(bb.minPoint.z*10) < TOL and
              abs(bb.maxPoint.z*10-mm('mp2T')) < TOL and
              abs(removed-math.pi*(diameter/2)**2*mm('mp2T')) < .1)
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
            bounds, [[mm('mp2X'), mm('mp2Y'), 0], [mm(tip), mm(top), mm('mp2T')]])):
        raise ValueError('Plate bounds differ from design parameters')
    report.update(single_connected_solid=True, bounds_mm=bounds, volume_mm3=geom.volume(comp))
    manager = f.TemporaryBRepManager.get()
    sections = []
    for x in (mm(right)+.01, mm(right)+mm('mp2RootR')+.01):
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
                ('thin-wall socket trial', 10, seat-34.2, seat-1),
                ('lower socket approach', 12, seat-34.2, seat-1)]:
            rad = od/2
            bb = [[x-rad,y-rad,start],[x+rad,y+rad,end]]
            env = manager.createCylinderOrCone(geom.point(x,y,start), rad/10,
                                               geom.point(x,y,end), rad/10)
            hits = []
            for body, bounds, path in bodies:
                if layout.overlaps(bb, bounds):
                    cut = manager.copy(env)
                    if not manager.booleanOperation(cut, body, f.BooleanTypes.IntersectionBooleanType):
                        raise ValueError('Cannot evaluate hardware space at hole ' + str(i+1))
                    if cut.lumps.count and cut.volume*1000 > .001:
                        hits.append(path)
            nearest = min(math.dist((x,y), p['center_xyz_mm'][:2]) for j,p in enumerate(points) if j != i)
            checks.append(dict(hole=i+1, envelope=label, od_mm=od, z_mm=[start,end],
                               interfering_bodies=hits,
                               lateral_margin_to_adjacent_washer_projection_mm=nearest-rad-4.5))
    return checks



def body_records(app, layout, part):
    return [(b, layout.box(app,b), b.assemblyContext.fullPathName + ' / ' + b.name)
            for b in layout.solids(part)]


def shifted_box(bounds, offset):
    return [[x, y+offset, z] for x,y,z in bounds]


def moving_copy(manager, body, dy):
    copy = manager.copy(body)
    transform = c.Matrix3D.create()
    transform.translation = c.Vector3D.create(0,dy/10,0)
    if not manager.transform(copy,transform):
        raise ValueError('Cannot position temporary camera geometry')
    return copy


def common_volume(manager, body, tool):
    target = manager.copy(body)
    if not manager.booleanOperation(target, tool, f.BooleanTypes.IntersectionBooleanType):
        raise ValueError('Temporary intersection could not be evaluated')
    return target.volume*1000 if target.lumps.count else 0


def inspect_motion(app, geom, layout, parts, plate, live, report, v5, v3):
    manager = f.TemporaryBRepManager.get()
    pb = report['bounds_mm']
    case = body_records(app,layout,parts['case'])
    camera = body_records(app,layout,parts['camera'])
    bracket = layout.find(parts['camera'],layout.BRACKET)
    bracket_path = bracket.fullPathName + ' / ' + bracket.bRepBodies.item(0).name
    usb_occ = layout.find(parts['camera'],layout.CAMERA)
    usb_path = usb_occ.fullPathName + ' / USB cable'
    native_seats = [face for face in bracket.component.bRepBodies.item(0).faces
                    if c.Plane.cast(face.geometry) and abs(face.geometry.normal.y)>.99999
                    and abs(face.pointOnFace.y*10)<TOL]
    face = max(native_seats,key=lambda face:face.area).createForAssemblyContext(bracket)
    seat = layout.box(app,face)
    points = live['mounting_points']
    for slot in points[4:]:
        ends = slot['arc_centers_xyz_mm']
        if (abs((slot['slot_length_mm']-4)/2-4)>TOL or
                abs(ends[1][0]-ends[0][0])>TOL or abs(ends[1][2]-ends[0][2])>TOL):
            raise ValueError('Unexpected slot travel or orientation')
    # Between x=140 and tip-R3 both tongue edges are straight or widened at the root.
    # The whole swept mating face lies in this guaranteed rectangular strip.
    tip = pb[1][0]
    root_x, upper = report['corner_radii_mm'][3]['xy']
    radius = report['corner_radii_mm'][1]['radius']
    margins = dict(lower_y=seat[0][1]-4-pb[0][1], upper_y=upper-seat[1][1]-4,
                   tip_x=tip-seat[1][0])
    backed = (min(margins.values())>=0 and seat[0][0]>=root_x and
              seat[1][0]<=tip-radius and max(abs(row[2]) for row in seat)<TOL)
    report['support'] = dict(mating_face_bounds_mm=seat, net_face_area_mm2=face.area*100,
        continuous_offsets_y_mm=[-4,4], swept_outer_footprint_supported=backed,
        minimum_edge_margins_mm=margins,
        contact_scope='Outer footprint checked continuously; net bearing excludes slots and plate bores. Maximum bore overhang occurs at either travel limit for these parallel symmetric slots.')
    report['plate_case_hits'] = [path for body,bb,path in case if layout.overlaps(bb,pb)
                                 and common_volume(manager,plate,body)>.001]
    report['interference_scope'] = dict(case_bodies=len(case), camera_bodies=len(camera),
        case_camera_hits=live['cross_assembly_interferences'], case_camera_x_gap_mm=live['measured_gap_mm'],
        unchanged_during_y_travel=True, excluded='Mounting_Plate_V1',
        internal_subassemblies='Rigid V5 and V3; internal checks remain those of the source assemblies.')
    lower = {'case':-v5['mounting_plate_interface']['material_sections_mm']['root_thickness'],
             'camera bracket':-v3['measured_thickness_mm']['vertical']}
    fixed = case + [(plate,pb,'Mounting_Plate_V2')]
    report['stations'] = []
    for dy in (-4,0,4):
        moving = []
        for original, bb, path in camera:
            body = moving_copy(manager,original,dy)
            predicted = shifted_box(bb,dy)
            actual = layout.box(app,body)
            if any(math.dist(a,b)>TOL for a,b in zip(actual,predicted)):
                raise ValueError('Temporary geometry frame mismatch: '+path)
            moving.append((body,actual,path))
        by_path = {path:body for body,bb,path in moving}
        imprint = manager.imprintOverlapBodies(by_path[bracket_path],plate,False)
        if not imprint[0]:
            raise ValueError('Cannot measure bracket bearing area')
        contact = sum(face_part.area*100 for face_part in imprint[3] if
                      c.Plane.cast(face_part.geometry) and abs(face_part.geometry.normal.z)>.99999
                      and abs(face_part.pointOnFace.z*10)<TOL)
        if not 0 < contact <= face.area*100+.02:
            raise ValueError('Invalid measured contact area')
        checks = hardware_checks(geom,layout,fixed+moving,points,pb[1][2],lower)
        report['stations'].append(dict(offset_y_mm=dy, bearing_area_mm2=contact,
            unsupported_net_face_area_mm2=max(0,face.area*100-contact),
            full_outer_face_supported=backed, usb_gap_mm=10*app.measureManager.measureMinimumDistance(
                by_path[usb_path],plate).value,
            plate_camera_hits=[path for body,bb,path in moving if layout.overlaps(bb,pb)
                               and common_volume(manager,body,plate)>.001], hardware=checks))
    # With equal bore/slot widths and collinear centres, a bore is entirely
    # inside its capsule up to the end-arc centre. Beyond that point the
    # uncovered crescent increases monotonically to the measured end position.
    report['support']['travel_with_bores_fully_inside_slots_each_way_mm'] = min(
        (s['slot_length_mm']-s['slot_width_mm'])/2 for s in points[4:])
    loss = max(s['unsupported_net_face_area_mm2'] for s in report['stations'])
    report['support'].update(max_net_area_loss_over_travel_mm2=loss,
        full_net_contact_over_travel_within_0_02_mm2=backed and loss<=.02)
    # Test the continuous swept bounding prism of each camera body against the actual
    # notched plate. A positive intersection is inconclusive between measured stations.
    candidates = []
    swept = []
    for body,bb,path in camera:
        box = [shifted_box(bb,-4)[0],shifted_box(bb,4)[1]]
        swept.append((path,box))
        if layout.overlaps(box,pb):
            center = geom.point(*(sum(pair)/2 for pair in zip(*box)))
            dims = [(b-a)/10 for a,b in zip(*box)]
            prism = manager.createBox(c.OrientedBoundingBox3D.create(center,layout.V(1,0,0),layout.V(0,1,0),*dims))
            if common_volume(manager,prism,plate)>.001:
                candidates.append(path)
    report['continuous_plate_camera_review_candidates'] = candidates
    report['usb_gap_lower_bound_over_travel_mm'] = max(0,min(
        s['usb_gap_mm'] for s in report['stations'])-2)
    report['usb_scope'] = 'Nearest sampled offset is within 2 mm everywhere. Real cable termination and bending remain unverified.'
    report['service_corridors'] = [dict(region=s['region'],bounds_mm=s['bounds'],
        plate_overlap=layout.overlaps(pb,s['bounds']),
        swept_camera_candidates=[path for path,bb in swept if layout.overlaps(bb,s['bounds'])])
        for s in live['service_access']['corridors']]
    report['lens'] = live['view_check']
    report['ventilation'] = dict(side_intake_clearance_mm=live['measured_gap_mm'],
        plate_above_case=pb[0][2]>=live['case_bounds'][1][2]-TOL,
        lid_faces_down=live['service_access']['ventilated_lid_faces_down'],thermal_test='pending')
    report['nominal_socket_review'] = [row for row in report['stations'][1]['hardware']
                                      if 'socket' in row['envelope'] and row['hole']<=4]
    report['assembly_method'] = ('Check the 10 mm OD socket trial at each hole and the adjacent washer margin. '
        'Where clear, hold the lower nut with a matching thin-wall socket and drive the upper head. '
        'Otherwise trial a slim open-ended wrench from the side; its swept handle and actual hardware need physical verification.')

def save_report(report):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=REPORT.parent,
                                     delete=False, suffix='.tmp') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    os.replace(stream.name,REPORT)


def build(app, report):
    design = f.Design.cast(app.activeProduct)
    if not design:
        design = f.Design.cast(app.activeDocument.products.itemByProductType('DesignProductType'))
    if not design:
        raise ValueError('Activate the assembly containing Mounting_Plate_V1.')
    root = design.rootComponent
    prior = [o for o in root.occurrences if o.component.name=='Mounting_Plate_V1']
    if len(prior)!=1 or any(o.component.name=='Mounting_Plate_V2' for o in root.allOccurrences):
        raise ValueError('Expected one root Mounting_Plate_V1 and no existing Mounting_Plate_V2.')
    old = prior[0]
    layout = helpers('CreateMountingLayoutV1')
    geom = helpers('CreateCameraMountV3')
    v1 = helpers('CreateMountingPlateV1')
    baseline = layout.read('mounting-layout-v1-report.json')
    old_report = layout.read('mounting-plate-v1-report.json')
    v3 = layout.read('camera-mount-v3-report.json')
    v5 = layout.read('jetson-case-v5-report.json')
    others = [o for o in root.occurrences if o!=old]
    if len(others)!=1 or root.bRepBodies.count:
        raise ValueError('Expected the V1 plate and the complete Mounting Layout occurrence in the active assembly.')
    parts = v1.verify_layout(others[0],layout,baseline)
    live = {}
    layout.inspect(app,design,parts,v5,v3,20,live)
    expected = [(6,38,0),(6,28,0),(126,38,0),(126,28,0),(172,9,0),(172,29,0)]
    points = live['mounting_points']
    if len(points)!=6 or any(math.dist(p['center_xyz_mm'],q)>TOL for p,q in zip(points,expected)):
        raise ValueError('Unexpected live mounting pattern')
    if old.bRepBodies.count!=1:
        raise ValueError('Expected a single original plate body')
    old_volume = geom.volume(old.component)
    if abs(old_volume-old_report['volume_mm3'])>.1:
        raise ValueError('V1 volume differs from its reference report')
    report.update(document=app.activeDocument.name, mating_faces=live['mounting_faces'],
        beam_clamp=dict(reserved_xy_mm=baseline['plate_proposal']['clamp_zone_xy'],
                       candidate_xy_mm=baseline['plate_proposal']['clamp_candidate_center_xy'], drilled=False))
    occ = None
    params = []
    visible = old.isLightBulbOn
    try:
        occ = root.occurrences.addNewComponent(c.Matrix3D.create())
        plate = create_plate(design,occ,geom,layout,points,report,params)
        report['size_mm'] = [b-a for a,b in zip(*report['bounds_mm'])]
        inspect_motion(app,geom,layout,parts,plate,live,report,v5,v3)
        comparison = report['comparison'] = dict(v1_volume_mm3=old_volume,v2_volume_mm3=report['volume_mm3'],
            delta_volume_mm3=report['volume_mm3']-old_volume,sections=[])
        for new in report['tongue_sections']:
            match, = [item for item in old_report['tongue_sections'] if abs(item['x_mm']-new['x_mm'])<TOL]
            comparison['sections'].append(dict(x_mm=new['x_mm'],v1_mm2=match['area_mm2'],v2_mm2=new['area_mm2'],
                                         delta_mm2=new['area_mm2']-match['area_mm2']))
        if not report['support']['swept_outer_footprint_supported'] or report['plate_case_hits'] or any(
                s['plate_camera_hits'] for s in report['stations']):
            raise ValueError('V2 support or plate interference check failed; inspect the V2 report.')
        report['status'] = 'review_required'
        old.isLightBulbOn = False
        report['v1_hidden'] = not old.isLightBulbOn
        save_report(report)
    except Exception:
        if old.isLightBulbOn != visible:
            old.isLightBulbOn = visible
        # Only objects created by this invocation can be rolled back.
        if occ and occ.isValid:
            occ.deleteMe()
        for param in reversed(params):
            if param.isValid:
                param.deleteMe()
        raise


def run(context):
    app = c.Application.get()
    report = dict(status='checking_active_assembly',units='mm',approved_for_installation=False,
                  strength_tested=False)
    try:
        build(app,report)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        save_report(report)
    message = report.get('error','').splitlines()
    app.userInterface.messageBox('\n'.join([report['status'],*message[-1:],str(REPORT)]))
