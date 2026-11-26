"""Round clamp CAD study. Dimensions and report in mm; beam axis Y, adapter bottom Z=0."""
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

import adsk.core as c
import adsk.fusion as f

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/'Documentation'
REPORT = DOCS/'round-beam-clamp-v1-report.json'
TITLE = 'VISION_SWINE_Round_Beam_Clamp_V1'
TOL = .01
NEW = f.FeatureOperations.NewBodyFeatureOperation
JOIN = f.FeatureOperations.JoinFeatureOperation
CUT = f.FeatureOperations.CutFeatureOperation
POS = f.ExtentDirections.PositiveExtentDirection

# The adapter is an existing mating part. Its measured seat and insert pattern
# are independent datums; changing clamp dimensions must not scale that part.
DEFAULTS = dict(rcBeamD=50, rcBeamLength=200, rcID=52, rcOD=68, rcWidth=40,
    rcFlangeWidth=110, rcLiner=1, rcGap=2, rcM8D=9, rcBoltX=40, rcBoltY=11,
    rcEarT=18, rcRootR=1.5, rcPadWidth=28, rcPadRise=9, rcPadOffset=3,
    rcSeat=47, rcM4X=7, rcM4D=4.5, rcCounterD=8, rcGrip=6, rcHeadD=7, rcHeadH=4,
    rcKeyD=3, rcM4Shank=4, rcM4InsertD=5.6, rcInsertL=10,
    rcBoltD=8, rcBoltL=50, rcAF=13, rcBoltHeadH=5.3, rcNutH=6.5,
    rcNutID=8.1, rcWasherD=16, rcWasherID=8.4, rcWasherT=1.6, rcSocketD=20,
    rcToolTravel=40)
DERIVED = dict(rcCentre='rcSeat+rcPadOffset+rcOD/2', rcTopSplit='rcCentre+rcGap/2',
    rcLowSplit='rcCentre-rcGap/2',rcTopEar='rcTopSplit+rcEarT',rcLowEar='rcLowSplit-rcEarT',
    rcScrewSeat='rcSeat+rcGrip',rcBoltSeat='rcTopEar+rcWasherT')


def helper(name):
    path=ROOT/'Scripts'/name/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_adapter(app):
    candidates=[]
    for doc in app.documents:
        design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
        if design:
            for occ in design.rootComponent.occurrences:
                if occ.component.name=='Adapter_V1' and occ.component.bRepBodies.count==1:
                    candidates.append((doc,occ.component.bRepBodies.item(0)))
    if len(candidates)!=1:
        raise ValueError('Open one original Adapter V1 document with root occurrence Adapter_V1.')
    return candidates[0]


def box_feature(p,w,depth,z,h,op,name):
    sk,pts=p.sketch(z,[('-('+w+')/2','-('+depth+')/2'),('('+w+')/2','('+depth+')/2')])
    sk.sketchCurves.sketchLines.addTwoPointRectangle(*pts)
    return p.g.extrude(p.comp,sk,h,POS,op,name)


def axial_cylinder(p,d,length,op,name):
    plane=p.comp.xZConstructionPlane
    ref=p.g.anchor(p.comp,plane,p.comp.yZConstructionPlane,p.plane('z','rcCentre'))
    sk=p.comp.sketches.add(plane)
    circle=sk.sketchCurves.sketchCircles.addByCenterRadius(p.g.linked_point(sk,ref),p.mm(d)/20)
    sk.sketchDimensions.addDiameterDimension(circle,p.g.point(p.mm(d),0)).parameter.expression=d
    features=p.comp.features.extrudeFeatures
    inp=features.createInput(sk.profiles.item(0),op)
    if not inp.setSymmetricExtent(p.g.value(length),True):
        raise ValueError('Cannot define full axial width: '+name)
    if op==CUT:
        inp.participantBodies=[p.comp.bRepBodies.item(0)]
    features.add(inp).name=name
    sk.isVisible=False


def root_fillet(p,upper):
    z=p.mm('rcTopEar' if upper else 'rcLowEar')
    x=math.sqrt(p.mm('rcOD/2')**2-(z-p.mm('rcCentre'))**2)
    edges=c.ObjectCollection.create()
    for e in p.comp.bRepBodies.item(0).edges:
        a,b=e.startVertex.geometry,e.endVertex.geometry
        if (abs(abs(a.x*10)-x)<TOL and abs(a.z*10-z)<TOL and
                abs(abs(a.y-b.y)*10-p.mm('rcWidth'))<TOL):
            edges.add(e)
    if edges.count!=2:
        raise ValueError('Expected two longitudinal flange root edges.')
    ff=p.comp.features.filletFeatures
    inp=ff.createInput()
    inp.edgeSetInputs.addConstantRadiusEdgeSet(edges,p.g.value('rcRootR'),False)
    ff.add(inp).name='Ear root fillets'


def half(p,upper,liner,stations):
    outer='rcBeamD+2*rcLiner' if liner else 'rcOD'
    axial_cylinder(p,outer,'rcWidth',NEW,'Round blank')
    if not liner:
        box_feature(p,'rcFlangeWidth','rcWidth','rcTopSplit' if upper else 'rcLowEar',
                    'rcEarT',JOIN,'Continuous flange')
        if not upper:
            box_feature(p,'rcPadWidth','rcWidth','rcSeat','rcPadRise',JOIN,'Adapter bearing pad')
    bottom='rcCentre-rcOD' if upper else 'rcLowSplit'
    height='rcOD+rcGap/2'
    box_feature(p,'rcFlangeWidth+rcOD','rcWidth+rcOD',bottom,height,CUT,'Split face')
    axial_cylinder(p,'rcBeamD' if liner else 'rcID','rcWidth+2 mm',CUT,'Beam-facing bore')
    if liner:
        return
    root_fillet(p,upper)
    for i,(x,y) in enumerate(stations,1):
        p.circle(x,y,'rcTopSplit' if upper else 'rcLowEar','rcM8D','rcEarT',CUT,f'M8 hole {i}')
    if not upper:
        for x in ('-rcM4X','rcM4X'):
            p.circle(x,'0','rcSeat','rcM4D','rcGrip',CUT,'M4 through land')
            p.circle(x,'0','rcScrewSeat','rcCounterD','rcCentre-rcScrewSeat',CUT,'Recessed head and key access')


def hexagon(p,x,y,z,height,op):
    radius=f'(rcAF/{math.sqrt(3):.15g})'
    coords=[(f'({x})+{math.cos(i*math.pi/3):.12f}*{radius}',
             f'({y})+{math.sin(i*math.pi/3):.12f}*{radius}') for i in range(6)]
    sk,pts=p.sketch(z,coords)
    for i,a in enumerate(pts):
        sk.sketchCurves.sketchLines.addByTwoPoints(a,pts[(i+1)%6])
    p.g.extrude(p.comp,sk,height,POS,op,'Hexagon across flats')


def ring(p,x,y,z,od,bore,h):
    p.circle(x,y,z,od,h,NEW,'Outside')
    p.circle(x,y,z,bore,h,CUT,'Bore')


def cylinder_faces(body,x,y,d):
    rows=[]
    for face in body.faces:
        cyl=c.Cylinder.cast(face.geometry)
        if cyl and abs(cyl.axis.z)>.99999 and math.dist([10*cyl.origin.x,10*cyl.origin.y,20*cyl.radius],[x,y,d])<TOL:
            bb=face.boundingBox
            rows.append([10*cyl.origin.x,10*cyl.origin.y,20*cyl.radius,10*bb.minPoint.z,10*bb.maxPoint.z])
    if len(rows)!=1:
        raise ValueError(f'Cannot identify unique cylindrical wall at {(x,y,d)}')
    return rows[0]


def close(a,b):
    return abs(a-b)<=TOL


def common(a,b):
    mgr=f.TemporaryBRepManager.get()
    temp=mgr.copy(a)
    if not mgr.booleanOperation(temp,b,f.BooleanTypes.IntersectionBooleanType):
        raise ValueError('Intersection calculation failed')
    return temp.volume*1000 if temp.lumps.count else 0.


def screw_length(grip,depth):
    lengths=(8,10,12,14,16,18,20,25,30)
    options=[length for length in lengths if 6<=length-grip+1e-8 and length-grip<=min(8,depth-1)+1e-8]
    if not options:
        raise ValueError('No trial M4 length gives 6–8 mm engagement and at least 1 mm bottom clearance.')
    return max(options)


def build(app,r):
    g=helper('CreateCameraMountV3')
    bc=helper('CreateBeamClampV1')
    layout=helper('CreateMountingLayoutV1')
    v3=layout.read('mounting-plate-v3-report.json')
    if v3['adapter_xy_height_mm']!=[97,45,47] or not v3['single_connected_solid']:
        raise ValueError('Expected the central V3 interface.')
    src,adapter=find_adapter(app)
    reference=layout.read('jetson-reference-recovery-report.json')['standalone_adapter']
    bb=layout.box(app,adapter)
    if (not adapter.isSolid or adapter.lumps.count!=1 or
            math.dist([b-a for a,b in zip(*bb)],reference['size_mm'])>TOL or
            abs(adapter.getPhysicalProperties(f.CalculationAccuracy.HighCalculationAccuracy).volume*1000
                -reference['volume_mm3'])>.1):
        raise ValueError('Adapter does not match inspected V1.')
    r['adapter_source']=src.name
    doc=app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
    doc.name=TITLE
    design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent!=f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent=f.DesignIntentTypes.HybridDesignIntentType
    design.designType=f.DesignTypes.ParametricDesignType
    for name,expr in {**{k:f'{v} mm' for k,v in DEFAULTS.items()},**DERIVED}.items():
        design.userParameters.add(name,g.value(expr),'mm','Round clamp prototype')
    parts={}
    def new(label):
        parts[label]=bc.Part(design,g,label)
        return parts[label]
    adapter_part=new('Adapter_V1_reference')
    base=adapter_part.comp.features.baseFeatures.add()
    if not base.startEdit():
        raise ValueError('Cannot edit reference base feature')
    try:
        stored=adapter_part.comp.bRepBodies.add(f.TemporaryBRepManager.get().copy(adapter),base)
        if not stored:
            raise ValueError('Cannot copy adapter')
    finally:
        base.finishEdit()
    stations=list(itertools.product(('-rcBoltX','rcBoltX'),('-rcBoltY','rcBoltY')))
    lower=new('Round_Beam_Lower_Clamp_V1')
    upper=new('Round_Beam_Upper_Clamp_V1')
    half(lower,False,False,stations)
    half(upper,True,False,stations)
    for name,is_upper in [('Upper_Rubber_Liner',True),('Lower_Rubber_Liner',False)]:
        half(new(name),is_upper,True,stations)
    axial_cylinder(new('Round_Beam_Reference_D50'),'rcBeamD','rcBeamLength',NEW,'Reference envelope')
    mm=lower.mm
    # Read the load path and pocket depths from the actual features before selecting hardware.
    stack=r['M4_stack']=[]
    choices=[]
    for x in (-mm('rcM4X'),mm('rcM4X')):
        wall=cylinder_faces(lower.body(),x,0,mm('rcM4D'))
        pocket=cylinder_faces(adapter_part.body(),x,0,mm('rcM4InsertD'))
        recess=cylinder_faces(lower.body(),x,0,mm('rcCounterD'))
        grip=wall[4]-wall[3]
        if not close(wall[3],pocket[4]) or not close(wall[4],recess[3]):
            raise ValueError('Adapter bearing surface or head seat mismatch')
        length=screw_length(grip,min(pocket[4]-pocket[3],mm('rcInsertL')))
        choices.append(length)
        stack.append(dict(clamp_bore=wall,adapter_pocket=pocket,counterbore=recess,
                          grip_mm=grip,length_mm=length,nominal_thread_overlap_mm=length-grip,
                          tip_to_pocket_bottom_mm=wall[4]-length-pocket[3]))
    if choices[0]!=choices[1]:
        raise ValueError('Asymmetric M4 stacks')
    design.userParameters.add('rcM4Length',g.value(f'{choices[0]} mm'),'mm','Selected from measured assembly')
    for i,x in enumerate(('-rcM4X','rcM4X'),1):
        p=new(f'M4_screw_{i}')
        p.circle(x,'0','rcScrewSeat-rcM4Length','rcM4Shank','rcM4Length',NEW,'Smooth shank')
        p.circle(x,'0','rcScrewSeat','rcHeadD','rcHeadH',JOIN,'Head envelope')
        ring(new(f'M4_insert_{i}'),x,'0','rcSeat-rcInsertL','rcM4InsertD','rcM4Shank','rcInsertL')
    bores=r['M8_axes_xy_d_z_mm']=[]
    for i,(x,y) in enumerate(stations,1):
        bores.append([cylinder_faces(p.body(),mm(x),mm(y),mm('rcM8D')) for p in (upper,lower)])
        bolt=new(f'M8_bolt_{i}')
        bolt.circle(x,y,'rcBoltSeat-rcBoltL','rcBoltD','rcBoltL',NEW,'Shank')
        hexagon(bolt,x,y,'rcBoltSeat','rcBoltHeadH',JOIN)
        for side,z in [('top','rcTopEar'),('bottom','rcLowEar-rcWasherT')]:
            ring(new(f'M8_washer_{side}_{i}'),x,y,z,'rcWasherD','rcWasherID','rcWasherT')
        nut=new(f'M8_nut_{i}')
        z='rcLowEar-rcWasherT-rcNutH'
        hexagon(nut,x,y,z,'rcNutH',NEW)
        nut.circle(x,y,z,'rcNutID','rcNutH',CUT,'Nut bore')
    if not design.computeAll():
        raise ValueError('Regeneration failed')
    inspect(app,layout,parts,r)
    r['parameters_mm']={p.name:10*p.value for p in design.userParameters}
    view=app.activeViewport
    camera=view.camera
    camera.viewOrientation=c.ViewOrientations.IsoTopRightViewOrientation
    camera.isFitView=True
    view.camera=camera
    view.fit()


def inspect(app,layout,parts,r):
    mm=parts['Round_Beam_Lower_Clamp_V1'].mm
    bodies={}
    for name,p in parts.items():
        p.body()  # checks one connected solid in each occurrence
        bodies[name]=p.comp.bRepBodies.item(0)  # all occurrences are at identity
    def distance_mm(a,b):
        # Each root occurrence contains exactly the one solid checked above.
        result=app.measureManager.measureMinimumDistance(parts[a].occ,parts[b].occ)
        if result is None or not math.isfinite(result.value) or result.value<0:
            raise ValueError(f'Invalid assembly distance for {a} and {b}')
        return result.value*10
    boxes={name:layout.box(app,b) for name,b in bodies.items()}
    r['solids']={k:dict(bounds_mm=boxes[k],size_mm=[b-a for a,b in zip(*boxes[k])],
                        volume_mm3=v.volume*1000) for k,v in bodies.items()}
    r['solid_count']=len(bodies)
    beam='Round_Beam_Reference_D50'
    lower='Round_Beam_Lower_Clamp_V1'
    upper='Round_Beam_Upper_Clamp_V1'
    adapter='Adapter_V1_reference'
    r['axial_surfaces_mm']={}
    dimension_checks=[]
    for name,required,width in [(beam,['rcBeamD'],'rcBeamLength'),
            (lower,['rcID','rcOD'],'rcWidth'),(upper,['rcID','rcOD'],'rcWidth'),
            ('Upper_Rubber_Liner',['rcBeamD','rcBeamD+2*rcLiner'],'rcWidth'),
            ('Lower_Rubber_Liner',['rcBeamD','rcBeamD+2*rcLiner'],'rcWidth')]:
        diameters=[]
        for face in bodies[name].faces:
            cyl=c.Cylinder.cast(face.geometry)
            if cyl and abs(cyl.axis.y)>.99999 and math.hypot(cyl.origin.x*10,cyl.origin.z*10-mm('rcCentre'))<TOL:
                diameters.append(cyl.radius*20)
        r['axial_surfaces_mm'][name]=diameters
        dimension_checks.extend(any(close(d,mm(expr)) for d in diameters) for expr in required)
        dimension_checks.append(close(r['solids'][name]['size_mm'][1],mm(width)))
    dimension_checks.extend(close(r['solids'][name]['size_mm'][0],mm('rcFlangeWidth')) for name in (lower,upper))
    r['main_dimensions_verified']=all(dimension_checks)
    mgr=f.TemporaryBRepManager.get()
    hits=r['unexpected_interferences_mm3']=[]
    intended=r['liner_interfaces']=[]
    for (a,ba),(b,bb) in itertools.combinations(bodies.items(),2):
        volume=0.
        if layout.overlaps(boxes[a],boxes[b]):
            volume=common(ba,bb)
        if frozenset((a,b)) in (frozenset((beam,'Upper_Rubber_Liner')),frozenset((beam,'Lower_Rubber_Liner')),
                                frozenset((upper,'Upper_Rubber_Liner')),frozenset((lower,'Lower_Rubber_Liner'))):
            distance=distance_mm(a,b)
            intended.append(dict(pair=[a,b],undeformed_overlap_mm3=volume,minimum_distance_mm=distance))
        elif volume>.001:
            hits.append(dict(pair=[a,b],volume_mm3=volume))
    imprint=mgr.imprintOverlapBodies(bodies[adapter],bodies[lower],False)
    if not imprint[0]:
        raise ValueError('Unable to imprint adapter seat')
    def seat_area(faces):
        return sum(face.area*100 for face in faces if c.Plane.cast(face.geometry)
                   and abs(face.geometry.normal.z)>.99999 and close(face.pointOnFace.z*10,mm('rcSeat')))
    available=seat_area(bodies[adapter].faces)
    contact=seat_area(imprint[3])
    r['adapter_bearing_mm2']=dict(available=available,measured=contact)
    tools=r['tool_paths']=[]
    def probe(name,x,y,z0,z1,d,targets):
        bbox=[[x-d/2,y-d/2,z0],[x+d/2,y+d/2,z1]]
        tool=mgr.createCylinderOrCone(c.Point3D.create(x/10,y/10,z0/10),d/20,
                                      c.Point3D.create(x/10,y/10,z1/10),d/20)
        clashes=[k for k in targets if layout.overlaps(bbox,boxes[k]) and common(tool,bodies[k])>.001]
        tools.append(dict(tool=name,xy_mm=[x,y],z_mm=[z0,z1],diameter_mm=d,hits=clashes))
    for i,(x,y) in enumerate(itertools.product((-mm('rcBoltX'),mm('rcBoltX')),(-mm('rcBoltY'),mm('rcBoltY'))),1):
        others=set(bodies)-{f'M8_bolt_{i}',f'M8_nut_{i}'}
        probe('M8 head socket',x,y,mm('rcBoltSeat'),mm('rcBoltSeat+rcBoltHeadH+rcToolTravel'),mm('rcSocketD'),others)
        probe('M8 nut socket',x,y,mm('rcLowEar-rcWasherT-rcNutH-rcToolTravel'),mm('rcLowEar-rcWasherT'),mm('rcSocketD'),others)
    for i,x in enumerate((-mm('rcM4X'),mm('rcM4X')),1):
        row=r['M4_stack'][i-1]
        shank=cylinder_faces(bodies[f'M4_screw_{i}'],x,0,mm('rcM4Shank'))
        insert=cylinder_faces(bodies[f'M4_insert_{i}'],x,0,mm('rcM4Shank'))
        row['measured_smooth_shank_overlap_mm']=min(shank[4],insert[4])-max(shank[3],insert[3])
        row['measured_tip_to_insert_bottom_mm']=shank[3]-insert[3]
        if (not close(row['measured_smooth_shank_overlap_mm'],row['nominal_thread_overlap_mm'])
                or not close(shank[4],row['counterbore'][3]) or row['measured_tip_to_insert_bottom_mm']<1-TOL):
            raise ValueError('Measured screw, insert or counterbore seat differs from stack')
        probe('M4 key before liner and beam',x,0,mm('rcScrewSeat+rcHeadH'),mm('rcCentre+rcToolTravel'),
              mm('rcKeyD'),[lower,adapter])
        row['screw_to_lower_liner_mm']=distance_mm(f'M4_screw_{i}','Lower_Rubber_Liner')
        row['screw_to_beam_mm']=distance_mm(f'M4_screw_{i}',beam)
    margins=[mm('(rcFlangeWidth-rcM8D)/2-rcBoltX'),mm('(rcWidth-rcM8D)/2-rcBoltY'),
             mm('2*rcBoltY-rcM8D'),mm('rcBoltX-rcM8D/2-rcID/2')]
    r['ear_material_mm']=dict(outer_x=margins[0],end_y=margins[1],between_holes=margins[2],
                              conservative_to_bore=margins[3],root_fillet=mm('rcRootR'))
    r['split_gap_mm']=boxes[upper][0][2]-boxes[lower][1][2]
    r['liner_radial_interference_mm']=mm('(rcBeamD+2*rcLiner-rcID)/2')
    protrusion=mm('rcLowEar-rcWasherT-rcNutH-(rcBoltSeat-rcBoltL)')
    r['bolt_tip_beyond_nut_mm']=protrusion
    passed=(all(dimension_checks) and not hits and all(not t['hits'] for t in tools) and min(margins)>0 and protrusion>0
            and close(r['split_gap_mm'],mm('rcGap')) and available>0 and abs(contact-available)<.1
            and all(row['minimum_distance_mm']<=TOL and row['undeformed_overlap_mm3']<=.001 for row in intended)
            and all(min(s['screw_to_lower_liner_mm'],s['screw_to_beam_mm'])>TOL for s in r['M4_stack']))
    r['status']='cad_prototype_review_required' if passed else 'geometry_review_required'


def run(context):
    app=c.Application.get()
    r=dict(status='started',document=TITLE,units='mm',approved_for_print=False,
           approved_for_installation=False,strength_tested=False,
           assembly_order=['Fasten lower clamp to adapter from inside the empty bore.',
                           'Fit liners and beam, then upper clamp, M8 washers, bolts and nuts.'],
           limits='CAD envelopes only. Metal load path, independent safety tether, rotation, slip and load testing required before field installation. Nominal rubber contact does not establish grip.')
    try:
        build(app,r)
    except Exception:
        r.update(status='incomplete',error=traceback.format_exc())
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=DOCS,delete=False) as out:
        json.dump(r,out,indent=2,allow_nan=False)
    os.replace(out.name,REPORT)
    app.userInterface.messageBox(r['status']+'\n'+str(REPORT))
