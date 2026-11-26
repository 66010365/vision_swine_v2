"""VISION SWINE V5. PCB datum in mm. Mounting plate seats against the underside."""
import importlib.util
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path

import adsk.core as c
import adsk.fusion as f

PROJECT = Path(__file__).resolve().parents[2]
DOCS = PROJECT / 'Documentation'
REPORT = DOCS / 'jetson-case-v5-report.json'


def build(app, report):
    # Use a private instance of the validated builder. Its checkpoints and log belong to V5.
    path = PROJECT / 'Scripts/CreateJetsonCaseV4/CreateJetsonCaseV4.py'
    spec = importlib.util.spec_from_file_location('vision_swine_v5_baseline', path)
    v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)
    v4.REPORT_PATH = REPORT
    v4.LOG_PATH = DOCS / 'jetson-case-v5-debug.log'
    v4.log('Starting V5 reference build')
    reference = v4.read('jetson-case-v4-report.json')
    v4.build(app, report)
    if report['status'] != 'review_required' or report['ventilation']['verified_openings'] != 11:
        raise RuntimeError('Base assembly did not pass its V4 geometry and ventilation checks')
    if report['step_sha256'] != reference['step_sha256'] or report['frame'] != reference['frame']:
        raise ValueError('Reference frame or STEP differs from inspected V4')
    report['v4_baseline'] = {key:report.pop(key) for key in (
        'interferences','cover_removal','panel_removal','vertical_board_removal','case_bounds','stages')}
    report.update(status='building_mounts', load_capacity='unverified; printed assembly requires load testing')
    design = f.Design.cast(app.activeProduct)
    root = design.rootComponent
    def occurrence(label):
        matches = [o for o in root.occurrences if o.component.name == label]
        if len(matches)!=1:
            raise ValueError('Expected one '+label)
        return matches[0]
    shell_occ = occurrence('Jetson_Nano_Case_V4')
    panel_occ = occurrence('Jetson_Nano_Front_Panel')
    cover_occ = occurrence('Jetson_Nano_Top_Cover')
    case = shell_occ.component
    case.name = 'Jetson_Nano_Case_V5'
    _, base = v4.previous_tools()
    origin = report['frame']['origin_assembly_mm']
    params = design.userParameters
    settings = dict(vsEarLength='20 mm',vsEarProjection='12 mm',vsEarThickness='5 mm',
        vsEarCenterY='50 mm',vsEarHolePitch='10 mm',vsEarHoleOffset='vsEarProjection/2',
        vsEarHoleDia='4.5 mm',vsEarRootR='1.5 mm',vsEarSeatZ='-vsPostH-vsFloor',
        vsEarTopZ='vsEarSeatZ+vsEarThickness',vsM4HeadOD='8 mm',vsM4HeadH='4 mm',
        vsM4NutEnvelopeOD='8.2 mm',vsM4NutH='3.2 mm',vsM4ToolOD='10 mm')
    for key,expression in settings.items():
        params.add(key,base.value(expression),'mm','V5 mounting interface; hardware envelopes are provisional')
    def mm(expr):
        return 10*design.unitsManager.evaluateExpression(expr,'mm')
    def sketch_at(level):
        inp=case.constructionPlanes.createInput()
        inp.setByOffset(case.xYConstructionPlane,base.value(level))
        plane=case.constructionPlanes.add(inp)
        plane.isLightBulbOn=False
        return case.sketches.add(plane)
    def positive(expr):
        return expr if mm(expr)>0 else f'-({expr})'
    def native_body():
        if case.bRepBodies.count!=1:
            raise ValueError('Ear must join the original case body')
        return case.bRepBodies.item(0)
    def volume():
        return 1000*native_body().getPhysicalProperties(f.CalculationAccuracy.HighCalculationAccuracy).volume

    y0='vsEarCenterY-vsEarLength/2'
    y1='vsEarCenterY+vsEarLength/2'
    seat,top=mm('vsEarSeatZ'),mm('vsEarTopZ')
    diameter=mm('vsEarHoleDia')
    thickness=top-seat
    reserve=reference['ventilation']['reserved_side_band_y_mm']
    if mm(y0)<reserve[0]-1e-6 or mm(y1)>reserve[1]+1e-6:
        raise ValueError('Ear outside reserved band')
    one_side=mm('vsEarHoleOffset')
    outer=mm('vsEarProjection')-one_side
    pitch=mm('vsEarHolePitch')
    end=(mm('vsEarLength')-pitch)/2
    fillet=mm('vsEarRootR')
    material=dict(end_edge=end-diameter/2,outer_edge=outer-diameter/2,
        wall_to_hole_edge=one_side-diameter/2,between_holes=pitch-diameter,
        hole_edge_to_fillet_toe=one_side-diameter/2-fillet,
        root_thickness=thickness,root_length=mm('vsEarLength'),root_fillet_radius=fillet)
    if min(material.values())<=0:
        raise ValueError('Nonpositive ear material section')
    interface=report['mounting_plate_interface']=dict(datum='PCB min X/Y and underside; +Z upward',
        normal=[0,0,-1],hole_axis=[0,0,1],plate_contact_z_mm=seat,ear_top_z_mm=top,
        material_sections_mm=material,assumed_parameters=settings,holes=[])
    ear_bounds=[]
    for side,sign,wall in [('minus_X',-1,'-vsMargin'),('plus_X',1,'vsPcbX+vsMargin')]:
        # One mm of overlap inside the existing side wall and floor gives a direct union.
        tip=f'({wall})+({sign})*vsEarProjection'
        inside=f'({wall})-({sign})*1 mm'
        xa,xb=sorted((tip,inside),key=mm)
        sk=sketch_at('vsEarTopZ')
        sk.name=side+' ear outline'
        lines=sk.sketchCurves.sketchLines.addTwoPointRectangle(base.point(mm(xa),mm(y0)),base.point(mm(xb),mm(y1)))
        corners=[p for line in lines for p in (line.startSketchPoint,line.endSketchPoint)]
        for x,y in [(xa,y0),(xb,y1)]:
            vertex=min(corners,key=lambda p:abs(p.geometry.x*10-mm(x))+abs(p.geometry.y*10-mm(y)))
            base.position(sk,vertex,positive(x),positive(y))
        base.extrude(case,sk.profiles.item(0),'vsEarThickness',f.FeatureOperations.JoinFeatureOperation,side+' base ear')
        edges=c.ObjectCollection.create()
        for edge in native_body().edges:
            if not c.Line3D.cast(edge.geometry):
                continue
            a,b=edge.startVertex.geometry,edge.endVertex.geometry
            if (all(abs(10*p.x-mm(wall))<0.001 and abs(10*p.z-top)<0.001 for p in (a,b))
                    and abs(min(a.y,b.y)*10-mm(y0))<0.001 and abs(max(a.y,b.y)*10-mm(y1))<0.001):
                edges.add(edge)
        if edges.count!=1:
            raise ValueError(side+' root edge not found uniquely')
        fillets=case.features.filletFeatures
        inp=fillets.createInput()
        inp.edgeSetInputs.addConstantRadiusEdgeSet(edges,base.value('vsEarRootR'),False)
        fillets.add(inp).name=side+' root reinforcement'
        ear_bounds.append((side,[[mm(xa),mm(y0),seat],[mm(xb),mm(y1),top+fillet]]))
        xc=f'({wall})+({sign})*vsEarHoleOffset'
        for station in (-1,1):
            yc=f'vsEarCenterY+({station})*vsEarHolePitch/2'
            x,y=mm(xc),mm(yc)
            sk=sketch_at('vsEarTopZ')
            circle=sk.sketchCurves.sketchCircles.addByCenterRadius(base.point(x,y),diameter/20)
            base.position(sk,circle.centerSketchPoint,positive(xc),yc)
            sk.sketchDimensions.addDiameterDimension(circle,base.point(x+3,y+3)).parameter.expression='vsEarHoleDia'
            before=volume()
            cut=base.extrude(case,sk.profiles.item(0),'vsEarThickness',f.FeatureOperations.CutFeatureOperation,
                             side+' M4 bore '+str(station))
            removed=before-volume()
            cylinders=[face.geometry for face in cut.faces if c.Cylinder.cast(face.geometry)]
            if not cylinders:
                raise ValueError('No cylindrical bore face')
            cyl=cylinders[0]
            measured=[cyl.origin.x*10,cyl.origin.y*10]
            actual_dia=cyl.radius*20
            if (math.dist(measured,(x,y))>0.001 or abs(actual_dia-diameter)>0.001 or abs(cyl.axis.z)<0.999999
                    or abs(removed-math.pi*(diameter/2)**2*thickness)>0.05
                    or any(native_body().pointContainment(base.point(x,y,z))!=f.PointContainment.PointOutsidePointContainment
                           for z in (seat+.01,top-.01))):
                raise ValueError('Through-bore verification failed')
            interface['holes'].append(dict(side=side,center_pcb_xy_mm=measured,
                assembly_xyz_at_plate_mm=[x+origin[0],y+origin[1],seat+origin[2]],
                diameter_mm=actual_dia,removed_mm3=removed,through=True))
    design.computeAll()
    parts={'case':shell_occ,'panel':panel_occ,'cover':cover_occ}
    report['solid_check']={}
    for label,occ in parts.items():
        ok=occ.bRepBodies.count==1 and occ.bRepBodies.item(0).isSolid and occ.bRepBodies.item(0).lumps.count==1
        report['solid_check'][label]=bool(ok)
        if not ok:
            raise ValueError(label+' is not a single connected solid')
    board=[b for b in root.bRepBodies if b.isSolid]
    for occ in root.allOccurrences:
        if occ.component not in [o.component for o in parts.values()]:
            board.extend(b for b in occ.bRepBodies if b.isSolid)
    board_boxes=[(b,base.bounds(app,b,origin)) for b in board]
    if not board_boxes:
        raise RuntimeError('Missing board solids')
    # V4 checked the complete original shell. Only the two ear boxes contain added material.
    # Carry forward the fresh baseline for board bodies outside that region.
    hits=report['interferences']={}
    for name,occ in parts.items():
        body=occ.bRepBodies.item(0)
        bounds=[bb for _,bb in ear_bounds] if name=='case' else [base.bounds(app,body,origin)]
        hits[name+'_board']=[base.name(b) for b,bb in board_boxes if any(base.overlap(bb,o) for o in bounds)
                             and base.interference(design,body,b)]
    for a,b in [('case','panel'),('case','cover'),('panel','cover')]:
        hits[a+'_'+b]=base.interference(design,parts[a].bRepBodies.item(0),parts[b].bRepBodies.item(0))
    report['case_bounds']=base.bounds(app,shell_occ.bRepBodies.item(0),origin)
    baseline=report['v4_baseline']
    report['case_board_check_basis']='Fresh V4 full assembly check plus exact pair tests in both added-material regions.'
    upward=max(0,baseline['vertical_board_removal']['travel_mm'])
    blocked=[base.name(b) for b,bb in board_boxes if any(base.overlap(
        [bb[0],[bb[1][0],bb[1][1],bb[1][2]+upward]],ear) for _,ear in ear_bounds)]
    pb=report['panel_bounds']
    rb=report['cover_bounds']
    report['removal_checks']=dict(
        cover_clear=baseline['cover_removal']['status']=='clear' and all(bb[1][2]<rb[0][2] for _,bb in ear_bounds),
        panel_clear=baseline['panel_removal']['status']=='clear' and all(bb[0][1]>=pb[1][1] for _,bb in ear_bounds),
        board_clear=baseline['vertical_board_removal']['with_panel']=='clear' and not blocked,
        board_sweep_candidates=blocked, conditions='Original V4 fastener and cable removal sequence; plate not modeled.')
    intake_high=mm('vsIntakeStart+2*(vsIntakeW+vsVentWeb)+vsIntakeW')
    vent_gap=mm(y0)-intake_high
    report['ear_vent_clearance']=dict(side_intake_y_gap_mm=vent_gap,
        cover_z_gap_mm=rb[0][2]-top-fillet,vents_verified_before_ears=report['ventilation']['verified_openings'],
        clear=vent_gap>0 and top+fillet<rb[0][2])
    # Provisional envelopes, checked with the screw entering from above and nut below the plate.
    # Model free approach from the whole lower half-space; plate thickness/hole relief are unknown.
    backgrounds=[(base.name(b),bb) for b,bb in board_boxes]
    backgrounds.extend((label,bb) for label,bb in report['mount_check']['boss_bounds'])
    backgrounds.extend([('front panel',pb),('main cover',[
        [mm('-vsMargin'),mm('-vsMargin'),rb[0][2]],[mm('vsPcbX+vsMargin'),mm('vsPcbY+vsMargin'),rb[1][2]]])])
    for label,bb in report['mount_check']['boss_bounds']:
        if 'cover boss' in label:
            backgrounds.append((label+' lid ear',[bb[0][:2]+[rb[0][2]],bb[1][:2]+[rb[1][2]]]))
    access=report['hardware_access']=dict(
        assumptions='One fastener serviced at a time, head above ear; compare envelopes with actual M4 hardware and future plate.',
        head_od_mm=mm('vsM4HeadOD'),head_height_mm=mm('vsM4HeadH'),nut_envelope_od_mm=mm('vsM4NutEnvelopeOD'),
        nut_height_mm=mm('vsM4NutH'),axial_tool_od_mm=mm('vsM4ToolOD'),holes=[])
    head,nut,tool=[mm(k)/2 for k in ('vsM4HeadOD','vsM4NutEnvelopeOD','vsM4ToolOD')]
    for hole in interface['holes']:
        x,y=hole['center_pcb_xy_mm']
        clearances=dict(head_edge=min(outer,end)-head,nut_edge=min(outer,end)-nut,
            head_to_fillet=one_side-fillet-head,
            driver_to_wall=one_side-tool-(fillet if top+mm('vsM4HeadH')<top+fillet else 0),
            driver_to_adjacent_head=pitch-tool-head,socket_to_adjacent_nut=pitch-tool-nut)
        obstruction=[]
        for label,r,low,high in [('head',head,top,top+mm('vsM4HeadH')),
                                ('upper tool',tool,top+mm('vsM4HeadH'),max(rb[1][2],max(bb[1][2] for _,bb in board_boxes))+1),
                                ('nut',nut,seat-mm('vsM4NutH'),seat),
                                ('lower tool',tool,seat-mm('vsM4NutH')-30,seat-mm('vsM4NutH'))]:
            box=[[x-r,y-r,low],[x+r,y+r,high]]
            obstruction.extend(label+' / '+part for part,bb in backgrounds if base.overlap(box,bb))
        access['holes'].append(dict(center_pcb_xy_mm=[x,y],clearances_mm=clearances,possible_obstructions=obstruction,
                                    clear=min(clearances.values())>=0 and not obstruction))
    report['verification_summary']=dict(
        single_solid_per_case_part=all(report['solid_check'].values()),
        four_verified_through_holes=len(interface['holes'])==4,
        vents_unobstructed=report['ear_vent_clearance']['clear'],
        provisional_hardware_envelopes_clear=all(h['clear'] for h in access['holes']),
        removal_paths_clear=all(report['removal_checks'][k] for k in ('cover_clear','panel_clear','board_clear')),
        no_component_interference=not any(hits.values()))
    report['hardware_fit_verified_with_real_parts']=False
    report['status']='interference_detected' if any(hits.values()) else 'review_required'
    app.activeViewport.fit()


def run(context):
    app=c.Application.get()
    names=('structure','fin_direction','roof_vents','left_intakes','right_intakes','interference','assembly_checks')
    report=dict(timestamp=datetime.now().isoformat(),units='mm',approved_for_print=False,status='started',
                scope='V5 initial as-built geometry; repeat inspection after edits.',
                stages={key:dict(status='not_run') for key in names})
    try:
        build(app,report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    finally:
        tmp=REPORT.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(tmp,REPORT)
    app.userInterface.messageBox(f"{report['status']}\n{REPORT}")

