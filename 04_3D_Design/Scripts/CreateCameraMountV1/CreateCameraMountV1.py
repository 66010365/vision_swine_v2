"""Standalone ELP L-bracket. CAD dimensions/report in mm; Fusion geometry in cm."""
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path

import adsk.core as c
import adsk.fusion as f

REPORT = Path(__file__).resolve().parents[2] / 'Documentation' / 'camera-mount-v1-report.json'
PARAMS = {
    'camWidth':'30 mm', 'camReach':'40 mm', 'camHeight':'50 mm', 'camT':'5 mm',
    'camRootR':'3 mm', 'camBodyWidth':'42 mm', 'camE':'12 mm', 'camF':'14 mm',
    'camRearOffset':'0 mm', 'camHoleD':'6.5 mm',
    'camHoleX':'(camWidth-camBodyWidth)/2+camF', 'camHoleY':'camRearOffset+camE',
    'camSlotL':'12 mm', 'camSlotW':'4.5 mm', 'camSlotPitch':'20 mm',
    'camSlotLowerZ':'13 mm', 'camSlotX':'camWidth/2',
}
value = c.ValueInput.createByString
TOL = 0.002


def point(x,y,z=0):
    return c.Point3D.create(x/10,y/10,z/10)


def offset(comp, plane, expr):
    inp = comp.constructionPlanes.createInput()
    if not inp.setByOffset(plane,value(expr)):
        raise ValueError('Cannot offset plane by '+expr)
    result = comp.constructionPlanes.add(inp)
    result.isLightBulbOn = False
    return result


def anchor(comp, sketch_plane, a, b):
    inp = comp.constructionPoints.createInput()
    if not inp.setByThreePlanes(sketch_plane,a,b):
        raise ValueError('Cannot locate sketch reference')
    result = comp.constructionPoints.add(inp)
    result.isLightBulbOn = False
    return result


def linked_point(sk, ref):
    projected = sk.project2([ref],True)
    if len(projected)!=1 or not f.SketchPoint.cast(projected[0]):
        raise ValueError('Expected a single linked sketch point')
    return projected[0]


def extrude(comp,sk,expr,direction,op,name):
    if sk.profiles.count != 1:
        raise ValueError(name+' requires one closed profile')
    features = comp.features.extrudeFeatures
    inp = features.createInput(sk.profiles.item(0),op)
    extent = f.DistanceExtentDefinition.create(value(expr))
    if not inp.setOneSideExtent(extent,direction):
        raise ValueError('Cannot set extent: '+name)
    if op == f.FeatureOperations.CutFeatureOperation:
        inp.participantBodies = [comp.bRepBodies.item(0)]
    feature = features.add(inp)
    feature.name = name
    sk.isVisible = False
    return feature


def volume(comp):
    return comp.bRepBodies.item(0).getPhysicalProperties(f.CalculationAccuracy.HighCalculationAccuracy).volume*1000


def build(app,report):
    doc = app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
    design = f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent == f.DesignIntentTypes.AssemblyDesignIntentType:
        design.designIntent = f.DesignIntentTypes.HybridDesignIntentType
    design.designType = f.DesignTypes.ParametricDesignType
    comp = design.rootComponent
    for key,expr in PARAMS.items():
        design.userParameters.add(key,value(expr),'mm','VISION SWINE | camera bracket V1')
    def mm(expr):
        return design.unitsManager.evaluateExpression(expr,'mm')*10

    width,reach,height,t,r = map(mm,('camWidth','camReach','camHeight','camT','camRootR'))
    hole_x,hole_y,d = map(mm,('camHoleX','camHoleY','camHoleD'))
    sx,z0,pitch,length,sw = map(mm,('camSlotX','camSlotLowerZ','camSlotPitch','camSlotL','camSlotW'))
    if min(t,r,d,sw,length-sw,pitch-length)<=0:
        raise ValueError('Invalid plate, hole, fillet or slot dimensions')
    xy = comp.xYConstructionPlane
    xz = comp.xZConstructionPlane
    def coord_plane(plane,axis,expr):
        sign = getattr(plane.geometry.normal,axis)
        if abs(abs(sign)-1)>1e-7:
            raise ValueError('Unexpected datum normal')
        return offset(comp,plane,expr if sign>0 else '-('+expr+')')
    def xplane(expr): return coord_plane(comp.yZConstructionPlane,'x',expr)
    def yplane(expr): return coord_plane(xz,'y',expr)
    def zplane(expr): return coord_plane(xy,'z',expr)

    width_plane = xplane('camWidth')
    top_plane = zplane('camHeight')
    pos,neg = f.ExtentDirections.PositiveExtentDirection,f.ExtentDirections.NegativeExtentDirection
    for plane,depth,extrusion,op,sense,title in [
        (xy,'camT','camHeight',f.FeatureOperations.NewBodyFeatureOperation,pos,'Mounting plate leg'),
        (top_plane,'camReach','camT',f.FeatureOperations.JoinFeatureOperation,neg,'Camera shelf')]:
        corner = anchor(comp,plane,width_plane,yplane(depth))
        sk = comp.sketches.add(plane)
        sk.sketchCurves.sketchLines.addTwoPointRectangle(sk.originPoint,linked_point(sk,corner))
        extrude(comp,sk,extrusion,sense,op,title)

    body = comp.bRepBodies.item(0)
    body.name = 'ELP_Camera_L_Bracket_V1'
    candidates = c.ObjectCollection.create()
    for edge in body.edges:
        if c.Line3D.cast(edge.geometry):
            a,b = edge.startVertex.geometry,edge.endVertex.geometry
            if (abs(edge.length*10-width)<TOL and
                all(abs(10*p.y-t)<TOL and abs(10*p.z-height+t)<TOL for p in (a,b))):
                candidates.add(edge)
    if candidates.count!=1:
        raise ValueError('Expected one inside corner at shelf underside')
    fillets = comp.features.filletFeatures
    fi = fillets.createInput()
    fi.edgeSetInputs.addConstantRadiusEdgeSet(candidates,value('camRootR'),False)
    fillet = fillets.add(fi)
    fillet.name = 'Inside corner reinforcement'
    radii = [face.geometry.radius*10 for face in fillet.faces if c.Cylinder.cast(face.geometry)]
    if not any(abs(rad-r)<TOL for rad in radii):
        raise ValueError('Corner fillet radius check failed')

    drill_ref = anchor(comp,top_plane,xplane('camHoleX'),yplane('camHoleY'))
    sk = comp.sketches.add(top_plane)
    circle = sk.sketchCurves.sketchCircles.addByCenterRadius(linked_point(sk,drill_ref),d/20)
    sk.sketchDimensions.addDiameterDimension(circle,point(hole_x+d,hole_y+d)).parameter.expression='camHoleD'
    before = volume(comp)
    drill = extrude(comp,sk,'camT',neg,f.FeatureOperations.CutFeatureOperation,'Camera screw clearance')
    def axes(feature,axis):
        result=[]
        for face in feature.faces:
            cyl=c.Cylinder.cast(face.geometry)
            if cyl and abs(getattr(cyl.axis,axis))>0.999999:
                coords=[10*v for v in cyl.origin.asArray()]
                pair=[coords[0],coords[1 if axis=='z' else 2]]
                if not any(math.dist(pair,p[:2])<TOL for p in result):
                    result.append(pair+[20*cyl.radius])
        return sorted(result,key=lambda p:p[1])
    actual = axes(drill,'z')
    removed = before-volume(comp)
    if (len(actual)!=1 or math.dist(actual[0],[hole_x,hole_y,d])>TOL or
            abs(removed-math.pi*(d/2)**2*t)>0.05):
        raise ValueError('Camera bore geometry mismatch')
    report['camera_bore']=dict(center_xyz_on_seat=[*actual[0][:2],height],diameter_mm=actual[0][2],
        axis=[0,0,1],removed_mm3=removed,reference_E=mm('camE'),reference_F=mm('camF'),
        camera_left_x=(width-mm('camBodyWidth'))/2,camera_rear_y=mm('camRearOffset'))
    slots=report['mounting_plate_slots']=[]
    xref=xplane('camSlotX')
    for i,zexpr in enumerate(('camSlotLowerZ','camSlotLowerZ+camSlotPitch'),1):
        z=mm(zexpr)
        refs=[anchor(comp,xz,xref,zplane(f'({zexpr}){sign}(camSlotL-camSlotW)/2'))
              for sign in ('-','+')]
        sk=comp.sketches.add(xz)
        ends=[linked_point(sk,ref) for ref in refs]
        sk.addCenterToCenterSlot(ends[0],ends[1],value('camSlotW'),True)
        # Choose the sense that cuts from the back mounting face, Y=0, toward +Y.
        n=sk.sketchToModelSpace(c.Point3D.create(0,0,1)).y-sk.sketchToModelSpace(c.Point3D.create(0,0,0)).y
        if abs(abs(n)-1)>1e-7:
            raise ValueError('Unexpected mounting plane')
        before=volume(comp)
        feature=extrude(comp,sk,'camT',pos if n>0 else neg,f.FeatureOperations.CutFeatureOperation,f'M4 slot {i}')
        ends=axes(feature,'y')
        expected=[[sx,z+sign*(length-sw)/2,sw] for sign in (-1,1)]
        area=(length-sw)*sw+math.pi*(sw/2)**2
        removed=before-volume(comp)
        if len(ends)!=2 or any(math.dist(a,b)>TOL for a,b in zip(ends,expected)) or abs(removed-area*t)>0.05:
            raise ValueError(f'Slot {i} size, location or volume mismatch')
        slots.append(dict(center_xyz=[(ends[0][0]+ends[1][0])/2,0,(ends[0][1]+ends[1][1])/2],
                          axis=[0,1,0],adjustment_axis=[0,0,1],end_arc_centers_xz_mm=[a[:2] for a in ends],
                          width=ends[0][2],overall_length=ends[1][1]-ends[0][1]+ends[0][2],removed_mm3=removed))
    design.computeAll()
    if comp.bRepBodies.count!=1 or not body.isSolid or body.lumps.count!=1:
        raise ValueError('Expected one connected solid')
    box=app.measureManager.getOrientedBoundingBox(body,c.Vector3D.create(1,0,0),c.Vector3D.create(0,1,0))
    measured=[10*v for v in (box.length,box.width,box.height)]
    sections=dict(camera_hole_to_side=min(hole_x,width-hole_x)-d/2,
        camera_hole_to_rear_or_front=min(hole_y,reach-hole_y)-d/2,
        camera_hole_to_inside_fillet=hole_y-d/2-t-r,
        slot_to_side=min(sx,width-sx)-sw/2,lower_slot_to_bottom=z0-length/2,
        upper_slot_to_inside_fillet=height-t-r-(z0+pitch+length/2),between_slots=pitch-length,
        plate_thickness=t,corner_fillet_radius=r)
    if math.dist(measured,[width,reach,height])>TOL or min(sections.values())<=0:
        raise ValueError('Outside dimensions or material clearances failed')
    # Compare the remaining exterior and interior planar faces after all cuts.
    levels={'y':[], 'z':[]}
    for face in body.faces:
        plane=c.Plane.cast(face.geometry)
        if plane:
            for axis in levels:
                if abs(getattr(plane.normal,axis))>0.999999:
                    levels[axis].append(10*getattr(face.pointOnFace,axis))
    def nearest(axis,target):
        return min(levels[axis],key=lambda coord:abs(coord-target))
    measured_t=[nearest('y',t)-nearest('y',0),nearest('z',height)-nearest('z',height-t)]
    if any(abs(actual_t-t)>TOL for actual_t in measured_t):
        raise ValueError('Remaining plate thickness mismatch')
    report['measured_thickness_mm']=dict(vertical=measured_t[0],horizontal=measured_t[1])
    report.update(status='review_required',single_connected_solid=True,measured_dimensions_xyz_mm=measured,
                  material_sections_mm=sections, mounting_face_y=0,angle_deg=90,parameter_expressions=PARAMS)
    app.activeViewport.fit()


def run(context):
    report=dict(timestamp=datetime.now().isoformat(),units='mm',status='started',approved_for_print=False,
                datum='Origin at left rear bottom; X across width, +Y toward lens/front, +Z upward.',
                assumptions='Camera centered across shelf; rear flush with outer mounting plane. F from left camera edge.',
                limitations=['Camera dimensions and handedness of F are provisional.',
                             'Confirm screw thread, length, head clearance and engagement on actual camera.',
                             'No measured load capacity or mounting plate assembly fit.'])
    try:
        build(c.Application.get(),report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    finally:
        REPORT.parent.mkdir(exist_ok=True)
        temp=REPORT.with_suffix('.tmp')
        temp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(temp,REPORT)
    c.Application.get().userInterface.messageBox(f"{report['status']}\n{REPORT}")

