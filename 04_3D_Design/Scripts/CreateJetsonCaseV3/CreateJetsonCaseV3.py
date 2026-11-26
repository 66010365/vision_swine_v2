"""Build V3 in a fresh STEP import. Feature geometry is local to the inspected PCB datum."""
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
NEW = f.FeatureOperations.NewBodyFeatureOperation
JOIN = f.FeatureOperations.JoinFeatureOperation
CUT = f.FeatureOperations.CutFeatureOperation


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
    source = PROJECT / 'Reference' / inspection['source']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = inspection['frame']
    if (any(r['frame'] != frame or r['step_sha256'] != digest for r in (v1report,v2report))
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
        params.add(key,base.value(expr),'mm','VISION SWINE V3')
    def mm(expr):
        return design.unitsManager.evaluateExpression(expr,'mm')*10
    def literal(number):
        return f'{number:.8f} mm'

    expressions = dict(v1report['initial_parameter_expressions'], vsWallH='39 mm',
        vsTop='vsWallH-vsPostH', vsPanelT='3 mm', vsPanelEar='2 mm', vsPortGapX='0.3 mm',
        vsPortGapZ='0.5 mm', vsBossSize='8 mm', vsBossWidth='3.8 mm', vsBossDepth='8 mm',
        vsPilot='2.5 mm', vsPilotDepth='6 mm', vsPanelScrew='3.4 mm')
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

    case_occ = component('Jetson_Nano_Case_V3')
    case = case_occ.component
    plate = base.create_plate(case)
    v2.add_walls(base,case,plate,mm('vsPcbX'),mm('vsPcbY'),mm('vsGap'),mm('vsMargin'))
    for i,(x,y) in enumerate(base.HOLES,1):
        base.add_post(case,i,x,y)
    panel_occ = component('Jetson_Nano_Front_Panel')
    panel = panel_occ.component
    back = '-vsMargin'
    front = '-vsMargin-vsPanelT'
    block(panel,'Front panel','-vsMargin-vsPanelEar','vsPcbX+vsMargin+vsPanelEar',front,back,'-vsPostH','vsTop',NEW)
    bosses = []
    for side,x0,x1,xc in [
        ('left','-vsMargin','-vsMargin+vsBossWidth','-vsMargin+vsBossWidth/2'),
        ('right','vsPcbX+vsMargin-vsBossWidth','vsPcbX+vsMargin','vsPcbX+vsMargin-vsBossWidth/2')]:
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
        if banks and left-banks[-1]['right'] < 1.0:
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
    design.computeAll()
    for occ in (case_occ,panel_occ):
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
    hits = report['interferences'] = {'case_board':[],'panel_board':[],
                                      'case_panel':base.interference(design,shell,cover)}
    # Conservative bounds cover all material, avoiding hundreds of unnecessary kernel checks.
    walls = [('floor',[cb[0],[cb[1][0],cb[1][1],-5]]),
             ('left wall',[[-4,-4,-5],[-1,83,34]]),('right wall',[[101,-4,-5],[104,83,34]]),
             ('rear wall',[[-4,80,-5],[104,83,34]])]
    for i,(x,y) in enumerate(base.HOLES,1):
        walls.append((f'post {i}',[[x-2.75,y-2.75,-5],[x+2.75,y+2.75,0]]))
    walls.extend(bosses)
    potential = []
    mount_candidates = []
    lift = cb[1][2]-min(bb[0][2] for _,bb in bodies)+1
    for body,bb in bodies:
        if any(base.overlap(bb,o) for _,o in walls) and base.interference(design,shell,body):
            hits['case_board'].append(base.name(body))
        if base.overlap(bb,pb) and base.interference(design,cover,body):
            hits['panel_board'].append(base.name(body))
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
        conditions='Continuous bounding-box sweep; unplug cables and remove board screws.')
    clear = min(bb[0][1] for _,bb in bodies) >= pb[1][1]-0.00001 and cb[0][1]>=pb[1][1]-0.00001
    report['panel_removal'] = dict(direction='-Y',status='clear' if clear and not hits['case_panel'] else 'review',
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
    report['status'] = 'interference_detected' if any(hits.values()) else 'review_required'
    app.activeViewport.fit()


def run(context):
    path = DOCS/'jetson-case-v3-report.json'
    report = dict(status='started',units='mm',timestamp=datetime.now().isoformat(),
                  scope='Initial V3 geometry only',approved_for_print=False)
    app = c.Application.get()
    try:
        build(app,report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    finally:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(tmp,path)
    app.userInterface.messageBox(f"{report['status']}\n{path}")



