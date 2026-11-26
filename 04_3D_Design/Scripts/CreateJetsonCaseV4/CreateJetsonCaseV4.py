"""Build V4 in a fresh STEP import. Feature geometry is local to the inspected PCB datum."""
import hashlib
import importlib.util
import json
import os
from math import pi
import traceback
from datetime import datetime
from pathlib import Path

import adsk
import adsk.core as c
import adsk.fusion as f

PROJECT = Path(__file__).resolve().parents[2]
DOCS = PROJECT / 'Documentation'
NEW = f.FeatureOperations.NewBodyFeatureOperation
JOIN = f.FeatureOperations.JoinFeatureOperation
CUT = f.FeatureOperations.CutFeatureOperation


REPORT_PATH = DOCS / 'jetson-case-v4-report.json'
LOG_PATH = DOCS / 'jetson-case-v4-debug.log'


def log(message):
    with LOG_PATH.open('a', encoding='utf-8') as stream:
        stream.write(f'{datetime.now().isoformat()} {message}\n')


def checkpoint(report):
    tmp = REPORT_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, REPORT_PATH)


def stage(report, name, action):
    record = report['stages'][name]
    record['status'] = 'running'
    log(f'{name}: started')
    checkpoint(report)
    try:
        action(record)
        record['status'] = 'passed'
    except Exception:
        record.update(status='failed', error=traceback.format_exc())
        log(record['error'])
    log(f"{name}: {record['status']}")
    checkpoint(report)
    return record

def read(name):
    return json.loads((DOCS / name).read_text(encoding='utf-8-sig'))


def previous_tools():
    path = PROJECT / 'Scripts/CreateJetsonCaseV2/CreateJetsonCaseV2.py'
    spec = importlib.util.spec_from_file_location('vision_swine_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.load_base_tools()


def build(app, report):
    v2, base = previous_tools()
    inspection = read('fusion-inspection.json')
    v1report = read('bottom-case-v1-report.json')
    v2report = read('jetson-case-v2-report.json')
    prior = read('jetson-case-v3-report.json')
    source = PROJECT / 'Reference' / inspection['source']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = inspection['frame']
    if (any(r['frame'] != frame or r['step_sha256'] != digest for r in (v1report,v2report,prior))
            or [frame[k] for k in 'xyz'] != [[1,0,0],[0,1,0],[0,0,1]]
            or not v2report['single_connected_solid']):
        raise ValueError('V1/V2 geometry references do not match')
    origin = frame['origin_assembly_mm']
    report.update(frame=frame,step_sha256=digest)
    mgr = app.importManager
    doc = mgr.importToNewDocument(mgr.createSTEPImportOptions(str(source)))
    if not doc:
        raise RuntimeError('Import failed')
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
    bodies = [(b,base.bounds(app,b,origin)) for b in solids]
    boards = [bb for b,bb in bodies if b.assemblyContext and
              b.assemblyContext.fullPathName == inspection['pcb_occurrence']]
    if not any(all(abs(a-b)<0.001 for side,key in zip(bb,('min','max'))
                   for a,b in zip(side,inspection['pcb'][key])) for bb in boards):
        raise ValueError('PCB datum mismatch')

    params = design.userParameters
    def add(key,expr):
        params.add(key,base.value(expr),'mm','VISION SWINE V4')
    def mm(expr):
        return design.unitsManager.evaluateExpression(expr,'mm')*10
    def literal(number):
        return f'{number:.8f} mm'

    expressions = dict(v1report['initial_parameter_expressions'], vsWallH='39 mm',
        vsTop='vsWallH-vsPostH', vsPanelT='3 mm', vsPanelEar='2 mm', vsPortGapX='0.3 mm',
        vsPortGapZ='0.5 mm', vsBossSize='8 mm', vsBossWidth='6 mm', vsBossDepth='8 mm',
        vsPilot='2.5 mm', vsPilotDepth='6 mm', vsPanelScrew='3.4 mm', vsFrontWeb='2 mm', vsLidT='3 mm', vsLidPad='7 mm',
        vsLidBossH='9 mm', vsLidPilot='2.5 mm', vsLidBore='3.4 mm', vsLidDepth='6 mm',
        vsSlotL='60 mm', vsSlotW='4 mm', vsVentWeb='3 mm', vsIntakeW='4 mm', vsIntakeH='12 mm',
        vsIntakeZ='9 mm', vsIntakeStart='18 mm', vsEarExtension='vsBossWidth-3.8 mm')
    for key,expr in expressions.items():
        add(key,expr)
    def component(label):
        matrix = c.Matrix3D.create()
        matrix.translation = c.Vector3D.create(*(v/10 for v in origin))
        occ = root.occurrences.addNewComponent(matrix)
        occ.component.name = label
        return occ

    def block(comp,title,x0,x1,y0,y1,z0,z1,op):
        # Dimension magnitudes are positive. Seed signs choose the side of the origin.
        inp = comp.constructionPlanes.createInput()
        inp.setByOffset(comp.xYConstructionPlane,base.value(z1))
        sk = comp.sketches.add(comp.constructionPlanes.add(inp))
        sk.name = title+' outline'
        xlo,xhi,ylo,yhi = map(mm,(x0,x1,y0,y1))
        lines = sk.sketchCurves.sketchLines.addTwoPointRectangle(base.point(xlo,ylo),base.point(xhi,yhi))
        corners = [p for line in lines for p in (line.startSketchPoint,line.endSketchPoint)]
        for x,y,ex,ey in [(xlo,ylo,x0,y0),(xhi,yhi,x1,y1)]:
            vertex = min(corners,key=lambda p:abs(p.geometry.x*10-x)+abs(p.geometry.y*10-y))
            base.position(sk,vertex,ex if x>0 else f'-({ex})',ey if y>0 else f'-({ey})')
        return base.extrude(comp,sk.profiles.item(0),f'({z1})-({z0})',op,title)

    def screw_cut(comp,title,xc,yfront,diameter,depth):
        # Resolve the XZ plane orientation from Fusion, including its offset sign.
        seed = comp.sketches.add(comp.xZConstructionPlane)
        n = seed.sketchToModelSpace(c.Point3D.create(0,0,1)).y
        seed.deleteMe()
        if abs(abs(n)-1)>1e-7:
            raise RuntimeError('Unexpected XZ plane orientation')
        plane = comp.constructionPlanes.createInput()
        plane.setByOffset(comp.xZConstructionPlane,base.value(yfront if n>0 else f'-({yfront})'))
        sk = comp.sketches.add(comp.constructionPlanes.add(plane))
        zc = 'vsTop-vsBossSize/2'
        center = sk.modelToSketchSpace(base.point(mm(xc),mm(yfront),mm(zc)))
        if abs(abs(center.x*10)-abs(mm(xc)))>0.001 or abs(abs(center.y*10)-mm(zc))>0.001:
            raise ValueError('Unexpected front sketch coordinate axes')
        circle = sk.sketchCurves.sketchCircles.addByCenterRadius(center,mm(diameter)/20)
        base.position(sk,circle.centerSketchPoint,xc if mm(xc)>0 else f'-({xc})',zc)
        sk.sketchDimensions.addDiameterDimension(circle,c.Point3D.create(center.x+.4,center.y+.3,0)).parameter.expression=diameter
        extrudes = comp.features.extrudeFeatures
        inp = extrudes.createInput(sk.profiles.item(0),CUT)
        direction = f.ExtentDirections.PositiveExtentDirection if n>0 else f.ExtentDirections.NegativeExtentDirection
        if not inp.setOneSideExtent(f.DistanceExtentDefinition.create(base.value(depth)),direction):
            raise RuntimeError('Cannot set screw bore extent')
        extrudes.add(inp).name=title

    case_occ = component('Jetson_Nano_Case_V4')
    case = case_occ.component
    plate = base.create_plate(case)
    v2.add_walls(base,case,plate,mm('vsPcbX'),mm('vsPcbY'),mm('vsGap'),mm('vsMargin'))
    for i,(x,y) in enumerate(base.HOLES,1):
        base.add_post(case,i,x,y)
    panel_occ = component('Jetson_Nano_Front_Panel')
    panel = panel_occ.component
    back = '-vsMargin'
    front = '-vsMargin-vsPanelT'
    block(panel,'Front panel','-vsMargin-vsEarExtension-vsPanelEar','vsPcbX+vsMargin+vsEarExtension+vsPanelEar',front,back,'-vsPostH','vsTop',NEW)
    bosses = []
    for side,x0,x1,xc in [
        ('left','-vsMargin-vsEarExtension','-vsMargin+vsBossWidth-vsEarExtension','-vsMargin-vsEarExtension+vsBossWidth/2'),
        ('right','vsPcbX+vsMargin-vsBossWidth+vsEarExtension','vsPcbX+vsMargin+vsEarExtension','vsPcbX+vsMargin+vsEarExtension-vsBossWidth/2')]:
        outside_lo, outside_hi = x0, x1
        block(case,side+' screw boss',outside_lo,outside_hi,back,'vsBossDepth-vsMargin',
              'vsTop-vsBossSize','vsTop',JOIN)
        screw_cut(case,side+' blind screw pilot',xc,back,'vsPilot','vsPilotDepth')
        screw_cut(panel,side+' screw clearance',xc,front,'vsPanelScrew','vsPanelT')
        bosses.append((side,[[mm(outside_lo),mm(back),mm('vsTop-vsBossSize')],
                             [mm(outside_hi),mm('vsBossDepth-vsMargin'),mm('vsTop')]]))

    known = {row['body'] for row in v1report['overhanging_bodies']}
    groups = {}
    found = set()
    for b,bb in bodies:
        if bb[0][1] >= -0.001:
            continue
        found.add(base.name(b))
        groups.setdefault(b.assemblyContext.fullPathName,[]).append(bb)
    if found != known or len(groups)!=6:
        raise ValueError('Front connector occurrences differ from the inspected STEP')
    openings = report['trial_port_openings'] = []
    # Closely spaced USB and Ethernet bodies share one aperture to avoid thin PETG webs.
    banks = []
    for path,boxes in sorted(groups.items(),key=lambda item:min(b[0][0] for b in item[1])):
        left = min(bb[0][0] for bb in boxes)-mm('vsPortGapX')
        if banks and left-banks[-1]['right'] < mm('vsFrontWeb'):
            banks[-1]['members'].append((path,boxes))
        else:
            banks.append(dict(members=[(path,boxes)]))
        banks[-1]['right'] = max(bb[1][0] for bb in boxes)+mm('vsPortGapX')
    for i,bank in enumerate(banks,1):
        boxes = [bb for _,members in bank['members'] for bb in members]
        prefix = f'vsPort{i}'
        edges = [min(b[0][0] for b in boxes),max(b[1][0] for b in boxes),
                 min(b[0][2] for b in boxes),max(b[1][2] for b in boxes)]
        for suffix,num in zip(('Xmin','Xmax','Zmin','Zmax'),edges):
            add(prefix+suffix,literal(num))
        a,b,d,e = (prefix+'Xmin-vsPortGapX',prefix+'Xmax+vsPortGapX',
                   prefix+'Zmin-vsPortGapZ',prefix+'Zmax+vsPortGapZ')
        block(panel,f'Connector opening {i}',a,b,front+'-1 mm',back+'+1 mm',d,e,CUT)
        for path,members in bank['members']:
            housing = [min(bb[0][0] for bb in members),max(bb[1][0] for bb in members),
                       min(bb[0][2] for bb in members),max(bb[1][2] for bb in members)]
            openings.append(dict(occurrence=path, aperture=i, housing_xz_mm=housing,
                front_recess_mm=min(bb[0][1] for bb in members)-mm(front),
                opening_xz_mm=list(map(mm,(a,b,d,e)))))
    lid_occ = component('Jetson_Nano_Top_Cover')
    lid = lid_occ.component
    lid_top = 'vsTop+vsLidT'
    slab = block(lid,'Rim-matched cover','-vsMargin','vsPcbX+vsMargin','-vsMargin',
                 'vsPcbY+vsMargin','vsTop',lid_top,NEW)
    upright = c.ObjectCollection.create()
    for edge in slab.bodies.item(0).edges:
        if abs(edge.startVertex.geometry.z-edge.endVertex.geometry.z)>0.001:
            upright.add(edge)
    if upright.count!=4:
        raise RuntimeError('Unexpected cover perimeter')
    fillets = lid.features.filletFeatures
    fil = fillets.createInput()
    fil.edgeSetInputs.addConstantRadiusEdgeSet(upright,base.value('vsCornerR'),False)
    fillets.add(fil).name='R3 cover perimeter'

    def xy_sketch(comp, height):
        inp=comp.constructionPlanes.createInput()
        inp.setByOffset(comp.xYConstructionPlane,base.value(height))
        return comp.sketches.add(comp.constructionPlanes.add(inp))

    def bore(comp,title,level,x,y,diameter,depth):
        sk=xy_sketch(comp,level)
        pt=base.point(mm(x),mm(y))
        circ=sk.sketchCurves.sketchCircles.addByCenterRadius(pt,mm(diameter)/20)
        base.position(sk,circ.centerSketchPoint,x if mm(x)>0 else f'-({x})',y)
        sk.sketchDimensions.addDiameterDimension(circ,c.Point3D.create(pt.x+.4,pt.y+.4,0)).parameter.expression=diameter
        base.extrude(comp,sk.profiles.item(0),depth,CUT,title)

    mount_checks=[]
    for side,x0,x1 in [('left','-vsGap-vsLidPad','-vsGap'),
                        ('right','vsPcbX+vsGap','vsPcbX+vsGap+vsLidPad')]:
        xc=f'(({x0})+({x1}))/2'
        for end,yc in [('front','12 mm'),('back','vsPcbY-10 mm')]:
            y0,y1=yc+'-vsLidPad/2',yc+'+vsLidPad/2'
            extent=[[mm(x0),mm(y0),mm('vsTop-vsLidBossH')],[mm(x1),mm(y1),mm('vsTop')]]
            conflicts=[label for label,bb in bosses if base.overlap(bb,extent)]
            board_candidates=[base.name(b) for b,bb in bodies if base.overlap(bb,extent)]
            title=side+' '+end
            mount_checks.append(dict(location=title,axis_xy=[mm(xc),mm(yc)],
                                     other_boss_overlaps=conflicts,board_candidates=board_candidates))
            if conflicts or board_candidates:
                raise ValueError('Occupied lid mount envelope: '+title)
            bosses.append((title+' cover boss',extent))
            block(case,title+' cover boss',x0,x1,y0,y1,'vsTop-vsLidBossH','vsTop',JOIN)
            block(lid,title+' cover ear',x0,x1,y0,y1,'vsTop',lid_top,JOIN)
            bore(case,title+' blind pilot','vsTop',xc,yc,'vsLidPilot','vsLidDepth')
            bore(lid,title+' cover bore',lid_top,xc,yc,'vsLidBore','vsLidT')
    report['cover_mounts']=mount_checks
    report['hardware']='Trial M3-size clearance, PETG pilot bores. Actual screw engagement and pull-out strength unverified.'

    report['stages']['structure']['status'] = 'passed'
    L,W,web=map(mm,('vsSlotL','vsSlotW','vsVentWeb'))
    center=[mm('vsPcbX')/2,mm('vsPcbY')/2]
    votes=[0.0,0.0]
    axis=0
    detected=False
    reason='Heatsink not located'
    sink_name=None
    try:
        sinks=[(b,bb) for b,bb in bodies if '110-0111-000' in base.name(b) and 'PORG_HSK_ASSY' in base.name(b)]
        if sinks:
            sink,sb=max(sinks,key=lambda pair:pair[0].volume)
            sink_name=base.name(sink)
            center=[(sb[0][k]+sb[1][k])/2 for k in range(2)]
            for face in sink.faces:
                plane=c.Plane.cast(face.geometry)
                if plane and face.boundingBox.maxPoint.z*10-origin[2]>(sb[0][2]+sb[1][2])/2:
                    for k,normal in enumerate((plane.normal.y,plane.normal.x)):
                        if abs(normal)>.999:
                            votes[k]+=face.area
            detected=max(votes)>0 and max(votes)>=1.4*min(votes)
            if detected:
                axis=votes.index(max(votes))
            reason='Dominant side-face area' if detected else 'No dominant fin-axis evidence'
    except Exception:
        reason=traceback.format_exc()
        log(reason)
    report['stages']['fin_direction']=dict(status='estimated' if detected else 'fallback_unconfirmed',
        initial_angle_deg=axis*90, side_area_votes_cm2=votes, heatsink=sink_name,reason=reason)
    log(f'fin direction: {reason}; votes={votes}; initial angle={axis*90} deg')
    ventilation=report['ventilation']=dict(fin_axis='XY'[axis] if detected else None,
        initial_slot_angle_deg=axis*90, angle_parameter='vsVentAngle',slot_length=L,slot_width=W,
        reserved_side_band_y_mm=[40,60],thermal_performance='unverified; physical temperature test required')

    def volume(comp):
        body=comp.bRepBodies.item(0)
        return body.getPhysicalProperties(f.CalculationAccuracy.HighCalculationAccuracy).volume*1000

    def verify_cut(comp,record,title,cut,area,thickness,sample):
        item=dict(name=title,volume_before_mm3=volume(comp),expected_removed_mm3=area*thickness)
        record['cuts'].append(item)
        body=comp.bRepBodies.item(0)
        inside=f.PointContainment.PointInsidePointContainment
        outside=f.PointContainment.PointOutsidePointContainment
        if body.pointContainment(sample)!=inside:
            raise ValueError(title+' center was not inside the intended solid before cut')
        try:
            cut()
        finally:
            item['volume_after_mm3']=volume(comp)
            item['removed_mm3']=item['volume_before_mm3']-item['volume_after_mm3']
            item['center_open']=comp.bRepBodies.item(0).pointContainment(sample)==outside
            log(f'{title}: {item}')
        if not item['center_open'] or abs(item['removed_mm3']-area*thickness)>0.05*thickness:
            raise ValueError(title+' failed the center or removed-volume check')
        record['verified_openings']+=1

    def verify_group(comp,record,count,thickness,create):
        record.update(expected_openings=count,verified_openings=0,cuts=[],volume_before_mm3=volume(comp))
        try:
            create()
        finally:
            record['volume_after_mm3']=volume(comp)
            record['removed_mm3']=record['volume_before_mm3']-record['volume_after_mm3']
            record['measured_open_mm2']=record['removed_mm3']/thickness
        if record['verified_openings']!=count:
            raise RuntimeError('Incomplete opening count')

    def make_roof(record):
        params.add('vsVentAngle',base.value(f'{axis*90} deg'),'deg',
                   'Prototype slot axis: 0 deg = PCB X, 90 deg = PCB Y')
        if not L>W>0 or web<3:
            raise ValueError('Invalid vent dimensions')
        # Keep both supported orientations inside the cover after a parameter edit.
        radius=max(L/2,2*(W+web)+W/2)
        for k,key in enumerate(('vsVentCX','vsVentCY')):
            low=-mm('vsMargin')+mm('vsCornerR')+web+radius
            high=mm(('vsPcbX','vsPcbY')[k])+mm('vsMargin')-mm('vsCornerR')-web-radius
            if low>high:
                raise ValueError('Both slot orientations must fit the cover')
            center[k]=min(high,max(low,center[k]))
            add(key,literal(center[k]))
        ventilation['pattern_center_xy']=center
        for i in range(5):
            # Rotate the row and each slot about the same center. Both end centers
            # are driven by expressions so changing 0 to 90 also moves the row.
            offset=f'({i-2})*(vsSlotW+vsVentWeb)'
            cx=f'vsVentCX-({offset})*sin(vsVentAngle)'
            cy=f'vsVentCY+({offset})*cos(vsVentAngle)'
            title=f'Rounded vent {i+1}'
            def cut():
                # Construct both endpoints before creating the sketch. Linked projections
                # follow parameter edits without driving dimensions on fixed sketch points.
                anchors=[]
                for sign in ('-','+'):
                    x=f'({cx}){sign}(vsSlotL-vsSlotW)/2*cos(vsVentAngle)'
                    y=f'({cy}){sign}(vsSlotL-vsSlotW)/2*sin(vsVentAngle)'
                    planes=[]
                    for plane,expr,component in [(lid.yZConstructionPlane,x,'x'),
                                                 (lid.xZConstructionPlane,y,'y')]:
                        normal=getattr(plane.geometry.normal,component)
                        if abs(abs(normal)-1)>1e-7:
                            raise ValueError(title+' unexpected local datum orientation')
                        inp=lid.constructionPlanes.createInput()
                        distance=expr if normal>0 else f'-({expr})'
                        if not inp.setByOffset(plane,base.value(distance)):
                            raise ValueError(title+' invalid endpoint plane')
                        datum=lid.constructionPlanes.add(inp)
                        datum.isLightBulbOn=False
                        planes.append(datum)
                    inp=lid.constructionPoints.createInput()
                    if not inp.setByThreePlanes(planes[0],planes[1],lid.xYConstructionPlane):
                        raise ValueError(title+' invalid endpoint planes')
                    anchor=lid.constructionPoints.add(inp)
                    anchor.isLightBulbOn=False
                    anchors.append((anchor,mm(x),mm(y)))

                sk=xy_sketch(lid,lid_top)
                sk.name=title+' profile'
                item=record['cuts'][-1]
                item.update(component=lid.name, target_body=lid.bRepBodies.item(0).name,
                            plane_z_mm=mm(lid_top), cut_distance_mm=mm('vsLidT'))
                try:
                    ends=[]
                    for anchor,x,y in anchors:
                        linked=sk.project2([anchor],True)
                        if len(linked)!=1 or not f.SketchPoint.cast(linked[0]):
                            raise ValueError(title+' endpoint projection failed')
                        pos=sk.sketchToModelSpace(linked[0].geometry)
                        if pos.distanceTo(base.point(x,y,mm(lid_top)))>0.0001:
                            raise ValueError(title+' endpoint outside expected lid coordinates')
                        ends.append(linked[0])
                    sk.addCenterToCenterSlot(ends[0],ends[1],base.value('vsSlotW'),True)
                    item['profile_count']=sk.profiles.count
                    if sk.profiles.count!=1:
                        raise ValueError(title+' expected one closed slot profile')
                    profile=sk.profiles.item(0)
                    actual_area=100*profile.areaProperties(f.CalculationAccuracy.HighCalculationAccuracy).area
                    item['profile_area_mm2']=actual_area
                    if profile.profileLoops.count!=1 or abs(actual_area-((L-W)*W+pi*W*W/4))>0.05:
                        raise ValueError(title+' slot profile area or loop count mismatch')
                    normal_z=(sk.sketchToModelSpace(c.Point3D.create(0,0,1)).z
                              -sk.sketchToModelSpace(c.Point3D.create(0,0,0)).z)
                    if abs(abs(normal_z)-1)>1e-7:
                        raise ValueError(title+' sketch normal is not parallel to lid Z')
                    direction=(f.ExtentDirections.NegativeExtentDirection if normal_z>0
                               else f.ExtentDirections.PositiveExtentDirection)
                    features=lid.features.extrudeFeatures
                    inp=features.createInput(profile,CUT)
                    extent=f.DistanceExtentDefinition.create(base.value('vsLidT'))
                    if not inp.setOneSideExtent(extent,direction):
                        raise ValueError(title+' cannot set through-cut extent')
                    inp.participantBodies=[lid.bRepBodies.item(0)]
                    feature=features.add(inp)
                    feature.name=title
                    # Sample inside both skins as well as mid-thickness (tested by verify_cut).
                    solid=lid.bRepBodies.item(0)
                    item['through_at_center']=all(solid.pointContainment(
                        base.point(mm(cx),mm(cy),mm(lid_top)-fraction*mm('vsLidT')))
                        ==f.PointContainment.PointOutsidePointContainment for fraction in (0.01,0.99))
                    if not item['through_at_center']:
                        raise ValueError(title+' residual material at lid skin')
                finally:
                    sk.isVisible=False
            verify_cut(lid,record,title,cut,(L-W)*W+pi*W*W/4,mm('vsLidT'),
                       base.point(mm(cx),mm(cy),mm('vsTop+vsLidT/2')))

    roof_result=stage(report,'roof_vents',lambda r:verify_group(lid,r,5,mm('vsLidT'),lambda:make_roof(r)))
    def make_side(record,side,x0,x1):
        for i in range(3):
            low=f'vsIntakeStart+{i}*(vsIntakeW+vsVentWeb)'
            high=low+'+vsIntakeW'
            extent=[[mm(x0),mm(low),mm('vsIntakeZ')],[mm(x1),mm(high),mm('vsIntakeZ+vsIntakeH')]]
            if mm(high)>40 or any(base.overlap(extent,b) for _,b in bosses):
                raise ValueError('Intake would enter mount keep-out')
            title=f'{side} intake {i+1}'
            def cut():
                block(case,title,x0,x1,low,high,'vsIntakeZ','vsIntakeZ+vsIntakeH',CUT)
            wall_x=mm('-vsMargin+vsWall/2' if side=='left' else 'vsPcbX+vsMargin-vsWall/2')
            verify_cut(case,record,title,cut,mm('vsIntakeW')*mm('vsIntakeH'),mm('vsWall'),
                       base.point(wall_x,(mm(low)+mm(high))/2,mm('vsIntakeZ+vsIntakeH/2')))

    for side,x0,x1 in [('left','-vsMargin-1 mm','-vsGap+0.2 mm'),
                        ('right','vsPcbX+vsGap-0.2 mm','vsPcbX+vsMargin+1 mm')]:
        stage(report,side+'_intakes',lambda r:verify_group(case,r,3,mm('vsWall'),
              lambda:make_side(r,side,x0,x1)))
    groups=[report['stages'][k] for k in ('roof_vents','left_intakes','right_intakes')]
    ventilation['verified_openings']=sum(g.get('verified_openings',0) for g in groups)
    if all(g['status']=='passed' for g in groups):
        ventilation.update(roof_open_mm2=groups[0]['measured_open_mm2'],
            side_open_mm2=sum(g['measured_open_mm2'] for g in groups[1:]),
            total_open_mm2=sum(g['measured_open_mm2'] for g in groups))
    case_bounds=base.bounds(app,case_occ.bRepBodies.item(0),origin)
    walls = [('floor',[case_bounds[0],[case_bounds[1][0],case_bounds[1][1],-5]]),
             ('left wall',[[-4,-4,-5],[-1,83,34]]),('right wall',[[101,-4,-5],[104,83,34]]),
             ('rear wall',[[-4,80,-5],[104,83,34]])]
    for i,(x,y) in enumerate(base.HOLES,1):
        walls.append((f'post {i}',[[x-2.75,y-2.75,-5],[x+2.75,y+2.75,0]]))
    walls.extend(bosses)
    hits=report['interferences']={}
    def check_pairs(record):
        parts={'case':case_occ,'panel':panel_occ,'cover':lid_occ}
        failures=record['errors']={}
        for first,second in [('case','panel'),('cover','case'),('cover','panel'),
                             ('case','board'),('panel','board'),('cover','board')]:
            key=first+'_'+second
            try:
                body=parts[first].bRepBodies.item(0)
                if second=='board':
                    bounds=[o for _,o in walls] if first=='case' else [base.bounds(app,body,origin)]
                    hits[key]=[base.name(b) for b,bb in bodies if any(base.overlap(bb,o) for o in bounds)
                               and base.interference(design,body,b)]
                else:
                    hits[key]=base.interference(design,body,parts[second].bRepBodies.item(0))
                log(f'{key}: {hits[key]}')
            except Exception:
                hits[key]=None
                failures[key]=traceback.format_exc()
                log(failures[key])
        if failures:
            raise RuntimeError('Incomplete interference checks: '+', '.join(failures))
    stage(report,'interference',check_pairs)
    report['stages']['assembly_checks']['status']='running'
    ap=[next(item['opening_xz_mm'] for item in openings if item['aperture']==a)
        for a in sorted({o['aperture'] for o in openings})]
    sections=dict(front_port_web=min(b[0]-a[1] for a,b in zip(ap,ap[1:])),
        front_bottom_strip=min(b[2]+mm('vsPostH') for b in ap),
        front_pilot_side=(mm('vsBossWidth')-mm('vsPilot'))/2,
        front_pilot_end=mm('vsBossDepth')-mm('vsPilotDepth'),
        cover_pilot_side=(mm('vsLidPad')-mm('vsLidPilot'))/2,
        cover_pilot_end=mm('vsLidBossH')-mm('vsLidDepth'),
        ear_bore_side=(mm('vsLidPad')-mm('vsLidBore'))/2,
        panel_hole_outer_edge=mm('vsPanelEar+vsBossWidth/2-vsPanelScrew/2'),
        roof_vent_web=web,intake_web=web,
        wall=mm('vsWall'),floor=mm('vsFloor'),cover=mm('vsLidT'),
        original_post_bore_side=(mm('vsPostOD')-mm('vsBore'))/2)
    if roof_result['status']=='passed':
        half=[2*(W+web)+W/2]*2
        half[axis]=L/2
        sections['cover_slot_edge']=min(min(center[k],mm(key)-center[k])+mm('vsMargin')-half[k] for k,key in enumerate(('vsPcbX','vsPcbY')))
    else:
        sections.pop('roof_vent_web')
    report['material_sections_mm']=sections
    report['minimum_of_checked_sections_mm']=min(sections.values())
    if sections['front_port_web'] < mm('vsFrontWeb')-0.001 or min(sections.values())<=0:
        raise ValueError('Insufficient material around front panel or mounts')
    design.computeAll()
    for occ in (case_occ,panel_occ,lid_occ):
        if occ.bRepBodies.count!=1 or not occ.bRepBodies.item(0).isSolid or occ.bRepBodies.item(0).lumps.count!=1:
            raise ValueError('Expected one connected solid in '+occ.component.name)
    shell,cover = [occ.bRepBodies.item(0) for occ in (case_occ,panel_occ)]
    cb,pb = [base.bounds(app,body,origin) for body in (shell,cover)]
    highest = max(bb[1][2] for _,bb in bodies)
    report.update(case_bounds=cb,panel_bounds=pb,total_height=cb[1][2]-cb[0][2],
                  clearance_above_assembly_mm=cb[1][2]-highest,
                  initial_parameters=expressions)
    if abs(report['total_height']-42)>0.001 or abs(report['clearance_above_assembly_mm']-5.724)>0.002:
        raise ValueError('Unexpected case height or assembly datum')
    if not (abs(pb[1][1]-cb[0][1])<0.001 and abs(pb[1][1]-pb[0][1]-mm('vsPanelT'))<0.001):
        raise ValueError('Panel/front datum mismatch')
    roof=lid_occ.bRepBodies.item(0)
    def material(body,x,y,z):
        p=base.point(x+origin[0],y+origin[1],z+origin[2])
        return body.pointContainment(p)==f.PointContainment.PointInsidePointContainment
    for mount in mount_checks:
        x,y=mount['axis_xy']
        top=mm('vsTop')
        checks=dict(cover_bore_open=not material(roof,x,y,top+mm('vsLidT')/2),
                    cover_bore_land=material(roof,x+mm('vsLidBore')/2+.15,y,top+mm('vsLidT')/2),
                    pilot_open=not material(shell,x,y,top-mm('vsLidDepth')/2),
                    pilot_land=material(shell,x+mm('vsLidPilot')/2+.15,y,top-mm('vsLidDepth')/2),
                    blind_end=material(shell,x,y,top-mm('vsLidDepth')-.2))
        mount['solid_samples']=checks
        if not all(checks.values()):
            raise ValueError('Cover attachment geometry failed: '+mount['location'])
    rb=base.bounds(app,roof,origin)
    gap=rb[0][2]-highest
    report.update(cover_bounds=rb,assembled_height_mm=rb[1][2]-cb[0][2],cover_clearance_above_assembly_mm=gap)
    report['cover_removal']=dict(direction='+Z',conditions='Remove four screws',
        status='clear' if rb[0][2]>=max(cb[1][2],pb[1][2],highest)-0.001 and
        hits['cover_case']==0 and hits['cover_panel']==0 and hits['cover_board']==[] else 'review')
    # Conservative bounds cover all material, avoiding hundreds of unnecessary kernel checks.
    potential = []
    mount_candidates = []
    lift = cb[1][2]-min(bb[0][2] for _,bb in bodies)+1
    for body,bb in bodies:
        for label,o in bosses:
            if base.overlap(bb,o):
                mount_candidates.append(dict(boss=label,body=base.name(body)))
        swept = v2.swept_box(bb,2,bb[1][2]+lift,1)
        for label,o in walls+[('front panel',pb)]:
            if base.overlap(swept,o):
                potential.append(dict(part=label,body=base.name(body)))
    report['mount_check'] = dict(boss_bounds=bosses,candidates=mount_candidates,
        method='Solid interference tested on joined case; screw tip must remain in blind pilot.')
    report['vertical_board_removal'] = dict(travel_mm=lift, potential_obstructions=potential,
        with_panel='review' if potential else 'clear',
        panel_removed='review' if any(p['part']!='front panel' for p in potential) else 'clear',
        conditions='Cover removed. Continuous bounding-box sweep; unplug cables and remove board screws.')
    clear = min(bb[0][1] for _,bb in bodies) >= pb[1][1]-0.00001 and cb[0][1]>=pb[1][1]-0.00001
    report['panel_removal'] = dict(direction='-Y',status='clear' if clear and hits['case_panel']==0 else 'review',
        method='All fixed solids behind seated rear plane throughout straight withdrawal; reverse for assembly.',
        conditions='Unplug peripherals and remove both front screws.')
    for op in openings:
        a,b,d,e = op['housing_xz_mm']
        limits = op['opening_xz_mm']
        corridor = [[a,pb[0][1]-1,d],[b,0,e]]
        possible = [label for label,o in walls if base.overlap(o,corridor)]
        op['housing_path_clear'] = limits[0]<a and limits[1]>b and limits[2]<d and limits[3]>e and not possible
        op['possible_case_obstacles'] = possible
    # The repeated occurrence of the same housing is the dual-stack USB candidate.
    usb = [op for op in openings if '065-0656-000_2:' in op['occurrence'].split('+')[-1]]
    accessory = []
    for op in usb:
        left,right,bottom,top = op['opening_xz_mm']
        accessory.append(dict(occurrence=op['occurrence'], aperture_width=right-left,
            aperture_height=top-bottom, trial_housing_recess=op['front_recess_mm'],
            status='Housing ID only. Compare actual accessories at their occupied socket heights.'))
    report['USB_accessory_fit_limits'] = dict(shared_stack_openings=accessory,
        horizontal_stack_pitch_mm=abs(sum(usb[0]['housing_xz_mm'][:2])-sum(usb[1]['housing_xz_mm'][:2]))/2 if len(usb)==2 else None,
        scope='Full aperture bounds per stack, not per socket. Plug nose reach is measured from actual receptacle face.')
    report['physical_access'] = dict(requested_ports=['DC power','HDMI','DisplayPort','USB Type-A','Ethernet','Micro USB'],
        basis='Six actual front occurrence envelopes; shared/stacked receptacles use a common opening. Functional labels unconfirmed.',
        USB_Wi_Fi='unverified; dongle outline, adjacent socket occupancy and projection needed',
        ELP_camera_USB='unverified; plug overmold, connector reach and bend radius needed',
        plug_fit='Trial openings. A housing-only corridor does not establish plug seating or removal.')
    report['stages']['assembly_checks']['status']='passed'
    if any(v['status']=='failed' for v in report['stages'].values()):
        report['status']='partial_failure'
    elif any(hits.values()):
        report['status']='interference_detected'
    else:
        report['status']='review_required'
    app.activeViewport.fit()


def run(context):
    names=('structure','fin_direction','roof_vents','left_intakes','right_intakes',
           'interference','assembly_checks')
    report=dict(status='started',units='mm',timestamp=datetime.now().isoformat(),
                scope='Initial V4 geometry only',approved_for_print=False,
                stages={name:dict(status='not_run') for name in names})
    app=c.Application.get()
    try:
        log('Starting V4 build')
        report['stages']['structure']['status']='running'
        checkpoint(report)
        build(app,report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
        for record in report['stages'].values():
            if record['status']=='running':
                record.update(status='failed',error=report['error'])
        log(report['error'])
    finally:
        checkpoint(report)
        log(f"Finished: {report['status']}")
    app.userInterface.messageBox(f"{report['status']}\n{REPORT_PATH}\n{LOG_PATH}")

