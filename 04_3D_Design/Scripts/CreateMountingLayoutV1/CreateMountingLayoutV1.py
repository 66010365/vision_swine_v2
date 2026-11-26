"""Layout only. World XY is the underside of a future overhead mounting plate."""
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path

import adsk.core as c
import adsk.fusion as f

DOCS = Path(__file__).resolve().parents[2] / 'Documentation'
REPORT = DOCS / 'mounting-layout-v1-report.json'
TOL = 0.01  # mm
CASE = 'Jetson_Nano_Case_V5'
PANEL = 'Jetson_Nano_Front_Panel'
LID = 'Jetson_Nano_Top_Cover'
BRACKET = 'Camera_L_Bracket_V3'
CAMERA = 'ELP-USB4K02AF-KL100W_reference'
HEAD = 'Camera_screw_head_reference'
V = c.Vector3D.create


def read(name):
    return json.loads((DOCS/name).read_text(encoding='utf-8-sig'))


def branches(parent):
    """Walk contextual occurrences; their body proxies are in the root assembly frame."""
    children = parent.childOccurrences if f.Occurrence.cast(parent) else parent.occurrences
    for child in children:
        yield child
        yield from branches(child)


def solids(parent):
    return [b for node in [parent, *branches(parent)] for b in node.bRepBodies if b.isSolid]


def find(parent, label):
    matches = [o for o in branches(parent) if o.component.name == label]
    if len(matches) != 1:
        raise ValueError(f'Expected one occurrence of {label}; found {len(matches)}')
    return matches[0]


def box(app,body):
    b = app.measureManager.getOrientedBoundingBox(body,V(1,0,0),V(0,1,0))
    center = [10*n for n in b.centerPoint.asArray()]
    half = [5*n for n in (b.length,b.width,b.height)]
    return [[x-y for x,y in zip(center,half)], [x+y for x,y in zip(center,half)]]


def union(boxes):
    return [[min(b[0][k] for b in boxes) for k in range(3)],
            [max(b[1][k] for b in boxes) for k in range(3)]]


def overlaps(a,b,axes=(0,1,2)):
    return all(min(a[1][i],b[1][i])-max(a[0][i],b[0][i]) > TOL for i in axes)


def unique_points(points):
    result=[]
    for p in points:
        if not any(math.dist(p,q)<TOL for q in result):
            result.append(p)
    return result


def face_point(face):
    return [10*n for n in face.pointOnFace.asArray()]


def mounting_plane(occ, axis, level):
    # Identify the large mating surface in native coordinates; read the corresponding root proxy.
    planes=[]
    for b in occ.component.bRepBodies:
        for face in b.faces:
            plane=c.Plane.cast(face.geometry)
            if plane and abs(plane.normal.asArray()[axis])>.99999 and abs(face_point(face)[axis]-level)<TOL:
                planes.append(face)
    native=max(planes,key=lambda face:face.area)
    proxy=native.createForAssemblyContext(occ)
    ok,n=proxy.evaluator.getNormalAtPoint(proxy.pointOnFace)
    if not ok:
        raise ValueError('Cannot measure mounting face normal')
    return dict(point_mm=face_point(proxy),outward_normal=n.asArray())


def bore_axis(occ, coords, axes, diameter, local_level, local_axis):
    """Find cylindrical walls in native coordinates; intersect their actual proxy axes with world Z=0."""
    points=[]
    for body in occ.component.bRepBodies:
        for face in body.faces:
            cyl=c.Cylinder.cast(face.geometry)
            if not cyl or abs(20*cyl.radius-diameter)>TOL or abs(cyl.axis.asArray()[local_axis])<.99999:
                continue
            if math.dist([10*cyl.origin.asArray()[i] for i in axes],coords)>TOL:
                continue
            bounds=face.boundingBox
            if not (10*bounds.minPoint.asArray()[local_axis]-TOL <= local_level <=
                    10*bounds.maxPoint.asArray()[local_axis]+TOL):
                continue
            actual=face.createForAssemblyContext(occ).geometry
            u=actual.axis.asArray()
            if abs(u[2])<.99999:
                raise ValueError('Fastener axis not perpendicular to mounting plane')
            p=[10*n for n in actual.origin.asArray()]
            points.append([p[0]-p[2]*u[0]/u[2],p[1]-p[2]*u[1]/u[2],0.0])
    matches=unique_points(points)
    if len(matches)!=1:
        raise ValueError(f'Cannot identify mounting axis at {coords}')
    return matches[0]


def inventory(parent):
    # Check all occurrences and BRep bodies, including surface bodies, after inserting the complete document.
    nodes=[parent,*branches(parent)]
    bodies=[b for n in nodes for b in n.bRepBodies]
    return dict(occurrences=len(nodes)-1,bodies=len(bodies),solids=sum(b.isSolid for b in bodies),
                faces=sum(b.faces.count for b in bodies))


def sources(app):
    found={}
    for doc in app.documents:
        design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
        if not design:
            continue
        labels={occ.component.name for occ in design.rootComponent.occurrences}
        for key,required in [('case',{CASE,PANEL,LID}),('camera',{BRACKET,CAMERA,HEAD})]:
            if required<=labels:
                if key in found:
                    raise ValueError(f'Multiple open {key} source assemblies; keep the intended original open.')
                found[key]=(doc,design.rootComponent)
    missing={'case','camera'}-found.keys()
    if missing:
        raise FileNotFoundError('Open the original full Fusion assemblies first: '+', '.join(sorted(missing)))
    for key,(doc,_) in found.items():
        if not doc.isSaved or doc.isModified:
            raise ValueError(f'Save the intended original {key} document in Fusion, then rerun: {doc.name}')
        file=doc.dataFile
        if not file or not file.isComplete or file.versionNumber!=file.latestVersionNumber:
            raise ValueError(f'Open the latest fully synced original {key} version: {doc.name}')
    return found


def matrix(translation, flip):
    m=c.Matrix3D.create()
    sign=-1 if flip else 1
    if not m.setWithCoordinateSystem(c.Point3D.create(*(n/10 for n in translation)),V(1,0,0),V(0,sign,0),V(0,0,sign)):
        raise ValueError('Cannot set assembly transform')
    return m


def interference(design,a,b):
    entities=c.ObjectCollection.create()
    entities.add(a)
    entities.add(b)
    query=design.createInterferenceInput(entities)
    query.areCoincidentFacesIncluded=False
    hits=design.analyzeInterference(query)
    if hits is None:
        raise ValueError('Fusion interference analysis failed')
    return hits.count


def build(app,r):
    v5=read('jetson-case-v5-report.json')
    v3=read('camera-mount-v3-report.json')
    src=sources(app)  # All source checks precede creation of a new document.
    case_root=src['case'][1]
    cam_root=src['camera'][1]
    signature={key:inventory(parent) for key,(_,parent) in src.items()}
    if not v5['verification_summary']['four_verified_through_holes'] or not v3['fastener_geometry_passed']:
        raise ValueError('Expected inspected V5/V3 source reports')
    # These are the original builders' root frames. Recheck the mating faces after insertion.
    seat=v5['mounting_plate_interface']['plate_contact_z_mm']
    plane1=mounting_plane(find(case_root,CASE),2,seat)
    plane2=mounting_plane(find(cam_root,BRACKET),1,0)
    if plane1['outward_normal'][2]>-.99999 or plane2['outward_normal'][2]<.99999:
        raise ValueError('Unexpected source orientation; inspect the original assembly version')
    r['sources']={k:dict(document=d.name,id=d.dataFile.id,version=d.dataFile.versionNumber,
                          source_inventory=signature[k]) for k,(d,_) in src.items()}
    answer,cancelled=app.userInterface.inputBox('Minimum X gap between full assembly envelopes (mm)',
                                                'Mounting Layout V1','20')
    if cancelled:
        r['status']='cancelled'
        return
    gap=float(answer)
    if not math.isfinite(gap) or gap<=0:
        raise ValueError('Gap must be a positive finite dimension in mm')
    cb=union([box(app,b) for b in solids(case_root)])
    mb=union([box(app,b) for b in solids(cam_root)])
    # Turn the complete case over. Its outward underside normal and the bracket mounting
    # normal now both point up; both mate to the underside of the same plate.
    case_t=[-cb[0][0],cb[1][1],plane1['point_mm'][2]]
    case_width=cb[1][0]-cb[0][0]
    case_depth=cb[1][1]-cb[0][1]
    cam_t=[case_width+gap-mb[0][0],case_depth/2-sum(b[1] for b in mb)/2,-plane2['point_mm'][2]]
    doc=app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
    doc.name='VISION SWINE Mounting Layout V1'
    try:
        design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
        if design.designIntent!=f.DesignIntentTypes.HybridDesignIntentType:
            design.designIntent=f.DesignIntentTypes.HybridDesignIntentType
        root=design.rootComponent
        inserted={}
        for key,trans,flip in [('case',case_t,True),('camera',cam_t,False)]:
            occ=root.occurrences.addByInsert(src[key][0].dataFile,matrix(trans,flip),False)
            if not occ:
                raise ValueError('Cannot insert complete '+key+' assembly')
            actual=inventory(occ)
            r['sources'][key].update(inserted_inventory=actual,transform=occ.transform2.asArray())
            if actual!=signature[key]:
                raise ValueError('Inserted assembly topology inventory differs: '+key)
            inserted[key]=occ
        inspect(app,design,inserted,v5,v3,gap,r)
        app.activeViewport.fit()
    except Exception:
        doc.close(False)  # Only discard the new incomplete layout, never a source document.
        raise


def inspect(app,design,inserted,v5,v3,gap,r,case_geometry=None):
    case=inserted['case']
    camera=inserted['camera']
    cc=find(case,CASE)
    bc=find(camera,BRACKET)
    interface=v5['mounting_plate_interface']
    seats=[mounting_plane(cc,2,interface['plate_contact_z_mm']),mounting_plane(bc,1,0)]
    if any(abs(p['point_mm'][2])>TOL or p['outward_normal'][2]<.99999 for p in seats):
        raise ValueError('Mounting faces do not share Z=0 with matching outward normals')
    r['mounting_faces']=dict(case=seats[0],bracket=seats[1])
    holes=[]
    for h in interface['holes']:
        p=bore_axis(cc,h['center_pcb_xy_mm'],(0,1),h['diameter_mm'],interface['plate_contact_z_mm'],2)
        holes.append(dict(part='case',center_xyz_mm=p,source_diameter_mm=h['diameter_mm']))
    for slot in v3['mounting_plate_slots']:
        ends=[bore_axis(bc,p,(0,2),slot['width'],0,1) for p in slot['end_arc_centers_xz_mm']]
        mid=[sum(v)/2 for v in zip(*ends)]
        holes.append(dict(part='camera bracket',center_xyz_mm=mid,arc_centers_xyz_mm=ends,
                          slot_width_mm=slot['width'],slot_length_mm=slot['overall_length']))
    def spacing(i,j):
        return math.dist(holes[i]['center_xyz_mm'],holes[j]['center_xyz_mm'])
    checks={'case_row_1':spacing(0,1),'case_row_2':spacing(2,3),
            'case_cross_1':spacing(0,2),'case_cross_2':spacing(1,3),'slot_pitch':spacing(4,5)}
    if any(abs(v-target)>TOL for v,target in zip(checks.values(),(10,10,120,120,20))):
        raise ValueError('Measured mounting pattern differs from V5/V3')
    r['measured_pitches_mm']=checks
    r['mounting_points']=holes
    from_case=[(b,box(app,b)) for b in (solids(case) if case_geometry is None else case_geometry)]
    from_camera=[(b,box(app,b)) for b in solids(camera)]
    cb,mb=[union([bb for _,bb in group]) for group in (from_case,from_camera)]
    overall=union([cb,mb])
    actual_gap=mb[0][0]-cb[1][0]
    if abs(actual_gap-gap)>TOL:
        raise ValueError('Inserted geometry does not match requested side gap')
    r.update(requested_gap_mm=gap,measured_gap_mm=actual_gap,case_bounds=cb,camera_bounds=mb,
             assembly_bounds=overall,assembly_size_mm=[b-a for a,b in zip(*overall)])
    # A positive separating-axis gap rules out interference for every cross-assembly pair.
    candidates=[(a,b) for a,ab in from_case for b,bb in from_camera if overlaps(ab,bb)]
    r['cross_assembly_interferences']=[dict(case_body=a.name,camera_body=b.name) for a,b in candidates
                                      if interference(design,a,b)]
    r['interference_scope']='All solid pairs across the two assemblies; disjoint bounds exclude collisions. Internal checks refer to source reports.'
    front=find(case,PANEL)
    roof=find(case,LID)
    cam=find(camera,CAMERA)
    lens=cam.component.bRepBodies.itemByName('Lens')
    if not lens:
        raise ValueError('Original lens reference missing')


    # Use end-cap centers in the inserted assembly, ordered by camera-local front direction.
    caps=[]
    for face in lens.faces:
        pl=c.Plane.cast(face.geometry)
        if pl and abs(pl.normal.y)>.99999:
            proxy=face.createForAssemblyContext(cam)
            caps.append((face_point(face)[1],proxy.centroid))
    if len(caps)!=2:
        raise ValueError('Cannot measure both lens end caps')
    caps.sort(key=lambda item:item[0])
    lens_dir=caps[0][1].vectorTo(caps[1][1])
    lens_dir.normalize()
    front_z=caps[1][1].z*10
    down_error=math.degrees(lens_dir.angleTo(V(0,0,-1)))
    if down_error>0.01:
        raise ValueError('Optical axis does not point down')
    ahead=[b.name for b,bb in from_case+from_camera if b.name!='Lens' and bb[0][2]<front_z-TOL]
    r['view_check']=dict(lens_direction=lens_dir.asArray(),angle_from_down_deg=down_error,
        lens_front_z_mm=front_z,forward_half_space_candidates=ahead,
        unobstructed_forward_half_space=not ahead,
        scope='Mechanical envelopes ahead of the lens front. Real field angle and working distance unmeasured.')
    # Proposed main rectangle plus camera tongue. Main extension on -X is a tentative clamp zone.
    edge=8.0
    main=[[cb[0][0]-30,cb[0][1]-edge],[cb[1][0]+edge,cb[1][1]+edge]]
    pts=[h['center_xyz_mm'] for h in holes[4:]]
    tongue=[[main[1][0],min(p[1] for p in pts)-edge],
            [max(p[0] for p in pts)+15,max(p[1] for p in pts)+edge]]
    if tongue[1][0]<=tongue[0][0]:
        raise ValueError('Cannot form camera mounting tongue')
    plate_bounds=union2(main,tongue)
    r['plate_proposal']=dict(shape='Union of two XY rectangles; no solid modeled',underside_z_mm=0,
        thickness_side='+Z',main_rectangle_xy=main,camera_tongue_xy=tongue,
        bounding_size_xy_mm=[b-a for a,b in zip(*plate_bounds)],
        nominal_slot_end_margin_mm=edge-(12/2),lens_forward_clear=front_z<0,
        clamp_zone_xy=[[main[0][0]+3,cb[0][1]+10],[cb[0][0]-3,cb[1][1]-10]],
        clamp_candidate_center_xy=[cb[0][0]-15,(cb[0][1]+cb[1][1])/2],
        clamp_status='Provisional; requires actual beam section, fasteners and load data')
    crossing=[]
    for b,bb in from_camera+from_case:
        if bb[1][2]>TOL and any(overlaps(bb,rect,(0,1)) for rect in (main,tongue)):
            crossing.append(b.name)
    r['plate_proposal']['plane_crossing_candidates']=crossing
    # Sweep outward, conservative boxes only; cable bodies/real plug sizes may be unavailable.
    panel_box=box(app,front.bRepBodies.item(0))
    lid_box=box(app,roof.bRepBodies.item(0))
    service=[]
    for label,bb,axis,end in [('panel and port access +Y',panel_box,1,cb[1][1]+30),
                              ('lid removal -Z',lid_box,2,cb[0][2]-30)]:
        corridor=[bb[0][:],bb[1][:]]
        if end<bb[0][axis]:
            corridor[0][axis]=end
        else:
            corridor[1][axis]=end
        service.append(dict(region=label,bounds=corridor,camera_candidates=[b.name for b,ob in from_camera if overlaps(ob,corridor)],
                            proposed_plate_overlap=any(overlaps(corridor,rect,(0,1)) for rect in (main,tongue)) and corridor[1][2]>TOL))
    r['service_access']=dict(corridors=service,physical_USB_and_power='30 mm service allowance; verify plugs, USB termination, bend radius and harness routing',
        side_intake_to_camera_x_gap_at_least_mm=actual_gap,ventilated_lid_faces_down=True,
        airflow='No thermal test; keep clamp and harness clear of openings')
    pb=r['plate_proposal']
    pb['hardware_holes']='Centers only; select plate clearance/tapped diameters for actual M4 fastening scheme.'
    r['review_flags']=dict(case_camera_collision=bool(r['cross_assembly_interferences']),
        possible_plate_material_conflict=bool(crossing),
        forward_view_needs_review=bool(ahead) or not pb['lens_forward_clear'],
        service_region_needs_review=any(s['camera_candidates'] or s['proposed_plate_overlap'] for s in service))
    r['status']='review_required'


def union2(a,b):
    return [[min(a[0][k],b[0][k]) for k in range(2)],[max(a[1][k],b[1][k]) for k in range(2)]]


def run(context):
    app=c.Application.get()
    r=dict(status='checking_sources',units='mm',approved_for_print=False,
           installation='Overhead common plate, both parts beneath; case rotated 180 deg about X.')
    try:
        build(app,r)
    except Exception:
        r.update(status='failed',error=traceback.format_exc())
    finally:
        tmp=REPORT.with_suffix('.tmp')
        tmp.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(tmp,REPORT)
    message=r.get('error','').splitlines()
    app.userInterface.messageBox(f"{r['status']}\n"+(message[-1]+'\n' if message else '')+str(REPORT))

