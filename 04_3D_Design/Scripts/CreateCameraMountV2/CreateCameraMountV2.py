"""Standalone ELP L-bracket. CAD dimensions/report in mm; Fusion geometry in cm."""
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path

import adsk.core as c
import adsk.fusion as f

REPORT = Path(__file__).resolve().parents[2] / 'Documentation' / 'camera-mount-v2-report.json'
PARAMS = {
    'camWidth':'30 mm', 'camReach':'40 mm', 'camHeight':'50 mm', 'camT':'5 mm',
    'camRootR':'3 mm', 'camBodyWidth':'42 mm', 'camBodyHeight':'42 mm', 'camBodyDepth':'36 mm',
    'camE':'12.5 mm', 'camF':'20 mm', 'camRefD':'6 mm', 'camPadH':'6 mm', 'camPadOD':'16 mm',
    'camLensOD':'18 mm','camLensL':'10 mm','camCableOD':'5 mm','camCableL':'35 mm',
    'camHeadOD':'10 mm','camHeadH':'4 mm','camDriverOD':'10 mm','camHoleD':'7 mm',
    'camHoleX':'camWidth/2', 'camHoleY':'12.5 mm',
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
    if design.designIntent != f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent = f.DesignIntentTypes.HybridDesignIntentType
    design.designType = f.DesignTypes.ParametricDesignType
    root = design.rootComponent
    # Ground datum is assembly XY. Local camera +Y (front) becomes world -Z.
    pose = c.Matrix3D.create()
    if not pose.setWithCoordinateSystem(point(0,0,0),c.Vector3D.create(1,0,0),
            c.Vector3D.create(0,0,-1),c.Vector3D.create(0,1,0)):
        raise RuntimeError('Cannot define downward-facing installation')
    bracket_occ = root.occurrences.addNewComponent(pose)
    comp = bracket_occ.component
    comp.name = 'Camera_L_Bracket_V2'
    for key,expr in PARAMS.items():
        design.userParameters.add(key,value(expr),'mm','VISION SWINE | camera bracket V2')
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
    body.name = 'ELP_Camera_L_Bracket_V2'
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
        camera_left_x=hole_x-mm('camF'),camera_rear_y=hole_y-mm('camE'))
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
    normals=[]
    for axis,target_level in [('y',0),('z',height)]:
        flat=next(face for face in body.faces if c.Plane.cast(face.geometry)
                  and abs(getattr(face.geometry.normal,axis))>.999999
                  and abs(10*getattr(face.pointOnFace,axis)-target_level)<TOL)
        normals.append(flat.geometry.normal)
    report['angle_deg']=math.degrees(math.acos(min(1,abs(normals[0].dotProduct(normals[1])))))
    if abs(report['angle_deg']-90)>0.01:
        raise ValueError('Bracket legs are not perpendicular')

    # Camera native origin: rear-left-bottom of HOUSING. Move the occurrence, not its bore.
    F,E,pad=map(mm,('camF','camE','camPadH'))
    local_offset=[hole_x-F,hole_y-E,height+pad]
    camera_origin=point(*local_offset)
    camera_origin.transformBy(bracket_occ.transform2)
    camera_pose=bracket_occ.transform2.copy()
    camera_pose.translation=c.Vector3D.create(*camera_origin.asArray())
    cam_occ=root.occurrences.addNewComponent(camera_pose)
    camera=cam_occ.component
    camera.name='ELP-USB4K02AF-KL100W_reference'
    def cp(plane,axis,expr):
        return offset(camera,plane,expr if getattr(plane.geometry.normal,axis)>0 else '-('+expr+')')
    X,Y,Z=camera.yZConstructionPlane,camera.xZConstructionPlane,camera.xYConstructionPlane

    def round_profile(plane,ref,diameter):
        sk=camera.sketches.add(plane)
        center=linked_point(sk,ref)
        circle=sk.sketchCurves.sketchCircles.addByCenterRadius(center,mm(diameter)/20)
        g=center.geometry
        sk.sketchDimensions.addDiameterDimension(circle,c.Point3D.create(g.x+.8,g.y+.7,0)).parameter.expression=diameter
        return sk

    # The approximate 6 mm projection is modeled normal to the camera underside.
    dx,dy=cp(X,'x','camF'),cp(Y,'y','camE')
    mount_center=anchor(camera,Z,dx,dy)
    new=f.FeatureOperations.NewBodyFeatureOperation
    foot=extrude(camera,round_profile(Z,mount_center,'camPadOD'),'camPadH',neg,new,'Reference mounting boss')
    foot.bodies.item(0).name='Mounting boss'
    socket=extrude(camera,round_profile(Z,mount_center,'camRefD'),'camPadH',neg,
                   f.FeatureOperations.CutFeatureOperation,'Nominal unthreaded bore')
    socket_axis=axes(socket,'z')
    if len(socket_axis)!=1 or math.dist(socket_axis[0],[F,E,mm('camRefD')])>TOL:
        raise ValueError('Camera local mounting bore mismatch')
    corner=anchor(camera,Z,cp(X,'x','camBodyWidth'),cp(Y,'y','camBodyDepth'))
    outline=camera.sketches.add(Z)
    outline.sketchCurves.sketchLines.addTwoPointRectangle(outline.originPoint,linked_point(outline,corner))
    housing=extrude(camera,outline,'camBodyHeight',pos,new,'Camera housing envelope')
    housing.bodies.item(0).name='Housing'
    middle_x=cp(X,'x','camBodyWidth/2')
    middle_z=cp(Z,'z','camBodyHeight/2')
    refs={}
    for label,plane,dia,span,sign in [
        ('Lens',cp(Y,'y','camBodyDepth'),'camLensOD','camLensL',1),
        ('USB cable',Y,'camCableOD','camCableL',-1)]:
        ref=anchor(camera,plane,middle_x,middle_z)
        sk=round_profile(plane,ref,dia)
        ny=sk.sketchToModelSpace(c.Point3D.create(0,0,1)).y-sk.sketchToModelSpace(point(0,0)).y
        if abs(abs(ny)-1)>1e-7:
            raise ValueError('Unexpected camera sketch orientation')
        feature=extrude(camera,sk,span,pos if sign*ny>0 else neg,new,label+' envelope')
        part=feature.bodies.item(0)
        part.name=label
        refs[label]=part
    design.computeAll()
    if camera.bRepBodies.count!=4 or any(not b.isSolid or b.lumps.count!=1 for b in camera.bRepBodies):
        raise ValueError('Incomplete camera reference geometry')

    def world(occ,xyz):
        p=point(*xyz)
        p.transformBy(occ.transform2)
        return p
    board_seat=world(bracket_occ,[*actual[0][:2],height])
    # Use the measured mounting surface and cylindrical axis in the camera frame.
    foot_body=camera.bRepBodies.itemByName('Mounting boss')
    seat_z=min(v.geometry.z for v in foot_body.vertices)*10
    camera_seat=world(cam_occ,[*socket_axis[0][:2],seat_z])
    normal_a=c.Vector3D.create(0,0,1)
    normal_b=normal_a.copy()
    normal_a.transformBy(bracket_occ.transform2)
    normal_b.transformBy(cam_occ.transform2)
    error=10*board_seat.distanceTo(camera_seat)
    report['assembly_alignment']=dict(camera_local_bore_xyz=[F,E,seat_z],
        camera_translation_in_bracket_mm=local_offset,
        seat_center_world_mm=[10*n for n in board_seat.asArray()],
        center_separation_mm=error,axis_parallel=normal_a.isParallelTo(normal_b))
    if error>TOL or not normal_a.isParallelTo(normal_b):
        raise ValueError('Camera bore and bracket bore are not coaxial at seat')

    bracket_body=bracket_occ.bRepBodies.item(0)
    contacts=report['camera_bracket_interference']={}
    for ref_body in cam_occ.bRepBodies:
        entities=c.ObjectCollection.create()
        entities.add(bracket_body)
        entities.add(ref_body)
        inp=design.createInterferenceInput(entities)
        inp.areCoincidentFacesIncluded=False
        result=design.analyzeInterference(inp)
        if result is None:
            raise RuntimeError('Missing interference result for '+ref_body.name)
        contacts[ref_body.name]=result.count
        if ref_body.name=='USB cable':
            distance=app.measureManager.measureMinimumDistance(bracket_body,ref_body)
            if distance is None:
                raise ValueError('Missing cable clearance result')
            report['usb_reference']=dict(minimum_bracket_gap_mm=10*distance.value,
                route='Straight stub from rear center toward world +Z',interferences=result.count,
                real_connector_bend_radius_and_strain_relief='unverified')

    lens_ys=[v.geometry.y*10 for v in refs['Lens'].vertices]
    lens_x,lens_z=mm('camBodyWidth')/2,mm('camBodyHeight')/2
    front=world(cam_occ,[lens_x,max(lens_ys),lens_z])
    back=world(cam_occ,[lens_x,min(lens_ys),lens_z])
    optical=back.vectorTo(front)
    optical.normalize()
    down_error=math.degrees(optical.angleTo(c.Vector3D.create(0,0,-1)))
    report['lens']=dict(direction_world=optical.asArray(),angle_from_vertical_down_deg=down_error,
                        ground_plane='Assembly XY',physical_lens_dimensions='illustrative')
    if down_error>0.01:
        raise ValueError('Lens axis is not downward')

    # The head starts flush under the shelf, so the maximum R intrusion is at the seat.
    # At depth q below the shelf the circular fillet reaches Y=t+r-sqrt(r^2-(r-q)^2).
    head_h=mm('camHeadH')
    q=min(head_h,r)
    y_limit=t+r-math.sqrt(max(0,r*r-(r-q)**2))
    head_gap=hole_y-mm('camHeadOD')/2-t-r
    driver_gap=hole_y-mm('camDriverOD')/2-y_limit
    report['fastener_access']=dict(method='Straight cylindrical head and axial driver, local -Z approach',
        assumed_head_diameter=mm('camHeadOD'),assumed_head_height=head_h,
        assumed_driver_diameter=mm('camDriverOD'),head_y_clearance_at_fillet=head_gap,
        driver_y_clearance_at_nearest_height=driver_gap,head_clear=head_gap>=0,driver_clear=driver_gap>=0,
        max_head_od_tangent_to_R3_mm=2*(hole_y-t-r),
        option_if_obstructed='Measure hardware first. For 0.5 mm gap use head OD <= %.2f mm or inside radius <= %.2f mm; recheck strength.'
        %(2*(hole_y-t-r-.5),hole_y-t-mm('camHeadOD')/2-.5))
    report.update(status='interference_detected' if any(contacts.values()) else 'review_required',
        physical_fit_approved=False,head_or_tool_review=head_gap<=0 or driver_gap<=0)
    app.activeViewport.fit()


def run(context):
    report=dict(timestamp=datetime.now().isoformat(),units='mm',status='started',approved_for_print=False,
                datum='Bracket-local origin at rear left bottom. World -Z points down; local +Y faces world -Z.',
                assumptions='42 wide x 42 high x 36 deep. F from camera left, E from rear. 6 mm foot projection normal to underside; OD 16 illustrative. Rear cable and centered lens are illustrative.',
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


