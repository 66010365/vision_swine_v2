"""VISION SWINE. Beam along Y, lower beam face Z=0. Report lengths in mm."""
import importlib.util
import itertools
import json
import math
import os
import tempfile
import traceback
from pathlib import Path

import adsk.core as c
import adsk.fusion as f

PROJECT = Path(__file__).resolve().parents[2]
DOCS = PROJECT / 'Documentation'
REPORT = DOCS / 'beam-clamp-v1-report.json'
PARAMS = {
    'bcBeamW':50, 'bcBeamH':50, 'bcBeamL':200, 'bcPlateW':110, 'bcPlateD':40,
    'bcPlateT':8, 'bcCornerR':3, 'bcPitchX':80, 'bcPitchY':22, 'bcHoleD':9,
    'bcRodD':8, 'bcRodL':130, 'bcNutAF':13, 'bcNutH':6.5, 'bcNutBore':8.1,
    'bcWasherOD':16, 'bcWasherID':8.4, 'bcWasherT':1.6, 'bcToolOD':20,
    'bcToolApproach':30, 'bcEndAllowance':2.5, 'bcSideAllowance':1,
}
NEW = f.FeatureOperations.NewBodyFeatureOperation
CUT = f.FeatureOperations.CutFeatureOperation
POS = f.ExtentDirections.PositiveExtentDirection
TOL = .01


def helper(name):
    path = PROJECT / 'Scripts' / name / (name+'.py')
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Part:
    """Each occurrence has an identity transform; its features follow assembly parameters."""
    def __init__(self, design, geom, name):
        self.design = design
        self.g = geom
        self.occ = design.rootComponent.occurrences.addNewComponent(c.Matrix3D.create())
        self.comp = self.occ.component
        self.comp.name = name
        self.planes = {}

    def mm(self, expr):
        return 10*self.design.unitsManager.evaluateExpression(expr,'mm')

    def plane(self, axis, expr):
        base = {'x':self.comp.yZConstructionPlane, 'y':self.comp.xZConstructionPlane,
                'z':self.comp.xYConstructionPlane}[axis]
        if expr == '0':
            return base
        key = (axis,expr)
        if key not in self.planes:
            signed = expr if getattr(base.geometry.normal,axis)>0 else '-('+expr+')'
            # A named offset remains associative even when a later edit makes it zero.
            self.planes[key] = self.g.offset(self.comp,base,signed)
        return self.planes[key]

    def sketch(self,z,coords):
        plane = self.plane('z',z)
        refs = [self.g.anchor(self.comp,plane,self.plane('x',x),self.plane('y',y)) for x,y in coords]
        sk = self.comp.sketches.add(plane)
        return sk,[self.g.linked_point(sk,ref) for ref in refs]

    def rect(self,w,d,z,height):
        sk,pts = self.sketch(z,[('-'+w+'/2','-'+d+'/2'),(w+'/2',d+'/2')])
        sk.sketchCurves.sketchLines.addTwoPointRectangle(*pts)
        self.g.extrude(self.comp,sk,height,POS,NEW,'Rectangular blank')

    def circle(self,x,y,z,d,h,op,title):
        sk,pts = self.sketch(z,[(x,y)])
        circ = sk.sketchCurves.sketchCircles.addByCenterRadius(pts[0],self.mm(d)/20)
        sk.sketchDimensions.addDiameterDimension(circ,self.g.point(self.mm(x)+8,self.mm(y)+6)).parameter.expression=d
        self.g.extrude(self.comp,sk,h,POS,op,title)

    def hex_nut(self,x,y,z):
        # Six vertices expressed using across-flats; two flats are parallel to X.
        af = 'bcNutAF'
        dx = af+f'/{math.sqrt(3):.15g}'
        vertices = [(dx,'0 mm'),(dx+'/2',af+'/2'),('-'+dx+'/2',af+'/2'),
                    ('-'+dx,'0 mm'),('-'+dx+'/2','-'+af+'/2'),(dx+'/2','-'+af+'/2')]
        sk,pts = self.sketch(z,[(f'({x})+({a})',f'({y})+({b})') for a,b in vertices])
        for i,p in enumerate(pts):
            sk.sketchCurves.sketchLines.addByTwoPoints(p,pts[(i+1)%6])
        self.g.extrude(self.comp,sk,'bcNutH',POS,NEW,'Hex nut reference')
        self.circle(x,y,z,'bcNutBore','bcNutH',CUT,'Thread envelope bore')

    def plate(self,z,stations):
        self.rect('bcPlateW','bcPlateD',z,'bcPlateT')
        edges=c.ObjectCollection.create()
        for edge in self.comp.bRepBodies.item(0).edges:
            if abs(edge.startVertex.geometry.z-edge.endVertex.geometry.z)*10>TOL:
                edges.add(edge)
        if edges.count!=4:
            raise ValueError('Expected four vertical plate edges')
        fillets=self.comp.features.filletFeatures
        inp=fillets.createInput()
        inp.edgeSetInputs.addConstantRadiusEdgeSet(edges,self.g.value('bcCornerR'),False)
        fillets.add(inp).name='Plate corners R'
        for i,(x,y) in enumerate(stations,1):
            self.circle(x,y,z,'bcHoleD','bcPlateT',CUT,f'M8 clearance {i}')

    def body(self):
        bodies=self.occ.bRepBodies
        if bodies.count!=1 or not bodies.item(0).isSolid or bodies.item(0).lumps.count!=1:
            raise ValueError('Expected one connected solid in '+self.comp.name)
        return bodies.item(0)


def model(design,g):
    parts={}
    def new(label):
        parts[label]=Part(design,g,label)
        return parts[label]
    stations=list(itertools.product(('-bcPitchX/2','bcPitchX/2'),('-bcPitchY/2','bcPitchY/2')))
    new('Beam_reference').rect('bcBeamW','bcBeamL','0','bcBeamH')
    new('Lower_plate').plate('-bcPlateT',stations)
    new('Upper_plate').plate('bcBeamH',stations)
    for i,(x,y) in enumerate(stations,1):
        new(f'Rod_{i}_ref').circle(x,y,'(bcBeamH-bcRodL)/2','bcRodD','bcRodL',NEW,'Smooth M8 rod envelope')
        for side,wz,nz in [('lower','-bcPlateT-bcWasherT','-bcPlateT-bcWasherT-bcNutH'),
                           ('upper','bcBeamH+bcPlateT','bcBeamH+bcPlateT+bcWasherT')]:
            washer=new(f'Washer_{side}_{i}_ref')
            washer.circle(x,y,wz,'bcWasherOD','bcWasherT',NEW,'Washer blank')
            washer.circle(x,y,wz,'bcWasherID','bcWasherT',CUT,'Washer bore')
            new(f'Nut_{side}_{i}_ref').hex_nut(x,y,nz)
    return parts,stations


def check_station(app,design,layout,parts,stations):
    mm=parts['Beam_reference'].mm
    bodies={k:p.body() for k,p in parts.items()}
    boxes={k:layout.box(app,b) for k,b in bodies.items()}
    beam=boxes['Beam_reference']
    h=mm('bcBeamH')
    def near(a,b):
        return abs(a-b)<TOL
    if any(not near(b-a,mm(expr)) for a,b,expr in zip(*beam,('bcBeamW','bcBeamL','bcBeamH'))):
        raise ValueError('Beam did not regenerate to requested dimensions')
    hits=[]
    for a,b in itertools.combinations(bodies,2):
        if layout.overlaps(boxes[a],boxes[b]) and layout.interference(design,bodies[a],bodies[b]):
            hits.append([a,b])
    manager=f.TemporaryBRepManager.get()
    plate_results={}
    for name,level in [('Lower_plate',0),('Upper_plate',h)]:
        body=bodies[name]
        size=[b-a for a,b in zip(*boxes[name])]
        if any(not near(n,mm(expr)) for n,expr in zip(size,('bcPlateW','bcPlateD','bcPlateT'))):
            raise ValueError(name+' size mismatch')
        holes=[]
        for face in body.faces:
            cyl=c.Cylinder.cast(face.geometry)
            if cyl and near(cyl.radius*20,mm('bcHoleD')) and abs(cyl.axis.z)>.99999:
                holes.append([cyl.origin.x*10,cyl.origin.y*10,cyl.radius*20,
                              (face.boundingBox.maxPoint.z-face.boundingBox.minPoint.z)*10])
        for x,y in stations:
            if sum(math.dist([mm(x),mm(y)],row[:2])<TOL and near(row[3],size[2]) for row in holes)!=1:
                raise ValueError(name+' through-hole centre or depth mismatch')
        if len(holes)!=4:
            raise ValueError('Expected exactly four bores')
        imprint=manager.imprintOverlapBodies(body,bodies['Beam_reference'],False)
        if not imprint[0]:
            raise ValueError('Cannot measure beam contact')
        area=sum(face.area*100 for face in imprint[3] if abs(face.pointOnFace.z*10-level)<TOL)
        plate_results[name]=dict(size_mm=size,bores_xy_diameter_depth_mm=holes,contact_mm2=area,
                                 expected_contact_mm2=mm('bcBeamW')*mm('bcPlateD'))
    tool_checks=[]
    for i,(xe,ye) in enumerate(stations,1):
        x,y=mm(xe),mm(ye)
        for side,sign in [('lower',-1),('upper',1)]:
            key=f'Nut_{side}_{i}_ref'
            nut=boxes[key]
            rod=boxes[f'Rod_{i}_ref']
            washer=boxes[f'Washer_{side}_{i}_ref']
            plate=boxes['Lower_plate' if sign==-1 else 'Upper_plate']
            inner,outer=(0,1) if sign==1 else (1,0)
            centred=all(math.dist([(bb[0][k]+bb[1][k])/2 for k in (0,1)],[x,y])<TOL
                        for bb in (rod,nut,washer))
            seated=(near(plate[outer][2],washer[inner][2]) and near(washer[outer][2],nut[inner][2]))
            if not centred or not seated or not near(rod[1][2]-rod[0][2],mm('bcRodL')):
                raise ValueError(f'Hardware stack or rod misaligned at {side} station {i}')
            # Start at nut bearing surface. The socket fits over its own nut and rod.
            start=nut[0 if sign==1 else 1][2]
            tip=nut[1 if sign==1 else 0][2]
            protrusion=sign*(rod[1 if sign==1 else 0][2]-tip)
            end=tip+sign*mm('bcToolApproach')
            radius=mm('bcToolOD')/2
            lo,hi=sorted((start,end))
            env=manager.createCylinderOrCone(c.Point3D.create(x/10,y/10,lo/10),radius/10,
                                             c.Point3D.create(x/10,y/10,hi/10),radius/10)
            box=[[x-radius,y-radius,lo],[x+radius,y+radius,hi]]
            obstacles=[]
            for label,body in bodies.items():
                if label in (key,f'Rod_{i}_ref') or not layout.overlaps(box,boxes[label]):
                    continue
                common=manager.copy(env)
                if not manager.booleanOperation(common,body,f.BooleanTypes.IntersectionBooleanType):
                    raise ValueError('Tool intersection could not be calculated')
                if common.lumps.count and common.volume*1000>.001:
                    obstacles.append(label)
            tool_checks.append(dict(rod=i,side=side,obstacles=obstacles,centred=centred,seated=seated,
                rod_projection_beyond_nut_mm=protrusion,
                minimum_blind_socket_internal_depth_mm=protrusion+mm('bcNutH'),
                nut_fully_on_rod_with_end_allowance=protrusion>=mm('bcEndAllowance')-TOL))
    ok=(not hits and all(not row['obstacles'] and row['nut_fully_on_rod_with_end_allowance'] for row in tool_checks)
        and all(near(p['contact_mm2'],p['expected_contact_mm2']) for p in plate_results.values()))
    return dict(beam_height_mm=h,beam_width_mm=mm('bcBeamW'),connected_solids=len(bodies),
                pair_count=math.comb(len(bodies),2),interferences=hits,plates=plate_results,
                tools_and_rod_ends=tool_checks,geometry_checks_passed=ok)

def interface_concept(lower):
    # Flush construction-only footprint on the outward face. No plate or beam holes.
    design=lower.design
    for key,value in [('bcInterfaceW','24 mm'),('bcInterfaceD','30 mm')]:
        design.userParameters.add(key,lower.g.value(value),'mm','Provisional adapter footprint')
    sk,pts=lower.sketch('-bcPlateT',[('-bcInterfaceW/2','-bcInterfaceD/2'),
                                      ('bcInterfaceW/2','bcInterfaceD/2')])
    sk.name='Concept only - lower adapter footprint'
    for line in sk.sketchCurves.sketchLines.addTwoPointRectangle(*pts):
        line.isConstruction=True
    sk.isVisible=False


def build(app,report):
    geom=helper('CreateCameraMountV3')
    layout=helper('CreateMountingLayoutV1')
    interface=layout.read('mounting-plate-v2-report.json')['beam_clamp']
    doc=app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
    doc.name='VISION SWINE Beam Clamp V1'
    design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent!=f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent=f.DesignIntentTypes.HybridDesignIntentType
    design.designType=f.DesignTypes.ParametricDesignType
    for name,number in PARAMS.items():
        design.userParameters.add(name,geom.value(f'{number} mm'),'mm','Beam Clamp V1 prototype')
    parts,stations=model(design,geom)
    interface_concept(parts['Lower_plate'])
    mm=parts['Beam_reference'].mm
    stack=mm('bcPlateT+bcWasherT+bcNutH')
    max_height=mm('bcRodL')-2*(stack+mm('bcEndAllowance'))
    rod_touch_width=mm('bcPitchX-bcRodD')
    width=rod_touch_width-2*mm('bcSideAllowance')
    report.update(datum='Beam centred in X and Y, beam spans Z=0..height; metal rods centred at beam mid-height.',
        parameters_mm=PARAMS,reference_hardware='Illustrative nut AF13 with smooth clearance bore; flat washers and smooth fully threaded rod envelopes. Confirm actual dimensions, thread engagement and hardware grade.',
        ranges=dict(requested_height_mm=[30,80],maximum_height_by_stack_and_end_allowance_mm=max_height,
            usable_height_interval_within_requested_mm=[30,min(80,max_height)],
            minimum_rod_length_at_height_80_mm=80+2*(stack+mm('bcEndAllowance')),
            width_at_rod_tangency_mm=rod_touch_width,required_clearance_per_side_mm=mm('bcSideAllowance'),
            maximum_width_with_side_allowance_mm=width,minimum_structural_width_mm=None,
            scope='Centred straight rectangular beam, positive width up to the clearance limit; narrower beams reduce contact area. No minimum load-bearing width established.'),
        plate_material_mm=dict(hole_to_x_edge=mm('(bcPlateW-bcPitchX-bcHoleD)/2'),
            hole_to_y_edge=mm('(bcPlateD-bcPitchY-bcHoleD)/2'),rod_to_hole_radial_clearance=mm('(bcHoleD-bcRodD)/2')),
        tool_assumption='Straight axial OD20 pass-through tool for AF13; verify actual tool bore, socket depth, ratchet swing and approach space on the installation.',
        mounting_interface=dict(lower_plate_datum_z_mm=-mm('bcPlateT'),construction_footprint_mm=[24,30],
            mounting_plate_v2_reserved_xy_mm=interface['reserved_xy_mm'],
            concept='Separate removable adapter between the central lower face and the reserved V2 area. Hardware and hole pattern deferred. The two coordinate systems are independent.',
            placement='Measure the horizontal assembled CG including wiring; locate beam reaction near its vertical projection and evaluate eccentric moment using actual mass. No final position or load rating.'),
        stations=[])
    height=design.userParameters.itemByName('bcBeamH')
    beam_w=design.userParameters.itemByName('bcBeamW')
    initial=(height.expression,beam_w.expression)
    try:
        for h,w in [(30,50),(50,50),(80,50),(50,width)]:
            height.expression=f'{h} mm'
            beam_w.expression=f'{w} mm'
            if not design.computeAll():
                raise ValueError('Fusion could not regenerate the test assembly')
            report['stations'].append(check_station(app,design,layout,parts,stations))
    finally:
        height.expression,beam_w.expression=initial
        if not design.computeAll():
            raise ValueError('Failed to restore nominal beam dimensions')
    report['geometry_checks_passed']=all(s['geometry_checks_passed'] for s in report['stations'])
    report['status']='review_required' if report['geometry_checks_passed'] else 'geometry_review_required'
    app.activeViewport.fit()


def run(context):
    app=c.Application.get()
    report=dict(status='creating',units='mm',approved_for_print=False,approved_for_installation=False,
                strength_tested=False)
    try:
        build(app,report)
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=DOCS,delete=False,suffix='.tmp') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    os.replace(stream.name,REPORT)
    message=report.get('error','').splitlines()
    app.userInterface.messageBox('\n'.join([report['status'],*message[-1:],str(REPORT)]))
