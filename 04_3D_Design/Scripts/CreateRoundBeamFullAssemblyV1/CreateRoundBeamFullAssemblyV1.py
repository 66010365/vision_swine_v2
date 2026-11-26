"""Round-beam installation study. All reported lengths are in assembly millimetres."""
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

import adsk.core as c
import adsk.fusion as f

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'Documentation'
REPORT=DOCS/'round-beam-full-assembly-v1-report.json'
TITLE='VISION_SWINE_Round_Beam_Full_Assembly_V1'
FULL='VISION_SWINE_Full_Assembly_V1'
ROUND='VISION_SWINE_Round_Beam_Clamp_V1'
BOARD='Jetson Nano B01 registered STEP'
FASTENERS='M4 fasteners and metal inserts (reference)'
REFERENCE='Adapter_V1_reference'
LOWER='Round_Beam_Lower_Clamp_V1'
UPPER='Round_Beam_Upper_Clamp_V1'
BEAM='Round_Beam_Reference_D50'
TOL=.01


def full_helpers():
    path=ROOT/'Scripts/CreateFullAssemblyV1/CreateFullAssemblyV1.py'
    spec=importlib.util.spec_from_file_location('full_assembly_helpers',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(condition,message):
    if not condition:
        raise ValueError(message)


def same_pose(a,b):
    check(max(abs(x-y) for x,y in zip(a.asArray(),b.asArray()))<1e-7,'Occurrence pose differs from reference')


def matching_root(app,title):
    docs=[doc for doc in app.documents if '_'.join(doc.name.split()).casefold()==title.casefold()]
    check(len(docs)==1,'Open exactly one '+title)
    design=f.Design.cast(docs[0].products.itemByProductType('DesignProductType'))
    check(design is not None,'Document has no design: '+title)
    return design.rootComponent


def reference_data(h,r):
    base=h.read('full-assembly-v1-report.json')
    clamp=h.read('round-beam-clamp-v1-report.json')
    recovery=h.read('jetson-reference-recovery-report.json')
    v3=h.read('mounting-plate-v3-report.json')
    step=ROOT/'Reference/Jetson_Nano_Dev_Kit_3D_b01.stp'
    check(base['document']==FULL and clamp['document']==ROUND and
          all(v['status']=='cad_prototype_review_required' for v in (base,clamp)) and
          not base['interference_hits'] and not clamp['unexpected_interferences_mm3'] and
          clamp['solid_count']==len(clamp['solids'])==26 and clamp['main_dimensions_verified'],
          'Expected the two completed and verified source reports')
    for field,path in [('recovery_sha256',DOCS/'jetson-reference-recovery-report.json'),
                       ('v3_report_sha256',DOCS/'mounting-plate-v3-report.json'),('board_step_sha256',step)]:
        check(h.digest(path)==base[field],'Changed verified reference: '+path.name)
    check(REFERENCE in clamp['solids'] and v3['adapter_xy_height_mm']==[97,45,47]
          and all(abs(clamp['parameters_mm'][k]-v)<TOL for k,v in
                  [('rcBeamD',50),('rcSeat',47),('rcM4Length',14),('rcBoltL',50)]),
          'Unexpected adapter or round hardware configuration')
    r['source_report_sha256']={name:h.digest(DOCS/name) for name in
        ('full-assembly-v1-report.json','round-beam-clamp-v1-report.json')}
    return base,clamp,recovery,v3,step


def hardware_names():
    keep={f'{name} {y}' for y in (20,70) for name in ('M4 flush insert','Flange washer','Flange M4x10')}
    keep.update(f'{name} {i}' for i in range(1,7) for name in
                ('M4 upper washer','M4 lower washer','M4 lower nut','Equipment M4 reference'))
    drop={f'{name} {x}' for x in (90,104) for name in ('Adapter upper insert','Clamp M4x12')}
    return keep,drop


def copy_bodies(h,src,dst,allowed=None):
    for b in src.bRepBodies:
        if allowed is not None and b.name not in allowed:
            continue
        stored=h.add_body(dst,f.TemporaryBRepManager.get().copy(b),b.name)
        check(stored.faces.count==b.faces.count and stored.edges.count==b.edges.count and
              abs(stored.volume-b.volume)*1000<.1,'Solid transfer mismatch: '+b.name)


def copy_tree(h,parent,source):
    dest=h.component(parent,source.component.name,source.transform2)
    copy_bodies(h,source.component,dest.component)
    for child in source.component.occurrences:
        copy_tree(h,dest.component,child)
    return dest


def names_match(collection,expected):
    actual=sorted(o.component.name for o in collection)
    check(actual==sorted(expected),'Component inventory mismatch: '+str(actual))


def retained(h,layout,root,src,r):
    groups={}
    signature={'plate':['Mounting_Plate_V3'],'adapter':['Adapter_V1'],
               'camera':[layout.CAMERA,layout.BRACKET,layout.HEAD],
               'case':[layout.CASE,layout.PANEL,layout.LID]}
    names_match(src.occurrences,[*(h.NAMES[k]+' assembly' for k in (*signature,'clamp')),BOARD,FASTENERS])
    poses=h.read('mounting-layout-v1-report.json')['sources']
    for key,children in signature.items():
        source=h.only(src,h.NAMES[key]+' assembly')
        names_match(source.component.occurrences,children)
        expected=h.matrix(poses[key]['transform']) if key in poses else (
            h.shift(97,45,6) if key=='adapter' else h.identity())
        same_pose(source.transform2,expected)
        groups[key]=copy_tree(h,root,source)
    src_hw=h.only(src,FASTENERS)
    same_pose(src_hw.transform2,h.identity())
    keep,drop=hardware_names()
    check(sorted(b.name for b in src_hw.component.bRepBodies)==sorted(keep|drop)
          and not src_hw.component.occurrences.count,'Equipment hardware inventory mismatch')
    groups['hardware']=h.component(root,FASTENERS,src_hw.transform2)
    copy_bodies(h,src_hw.component,groups['hardware'].component,keep)
    check(groups['hardware'].component.bRepBodies.count==len(keep),'Incomplete retained hardware')
    r['exclusions']=[h.NAMES['clamp']+' assembly',REFERENCE,*sorted(drop)]
    return groups


def registered_board(app,h,layout,src,root,rec,base,step,r):
    source=h.only(src,BOARD)
    pose=source.transform2
    same_pose(pose,h.matrix(rec['registration']['matrix_translation_cm']))
    pcb_path=h.read('fusion-inspection.json')['pcb_occurrence']
    def carrier(group):
        matches=[o for o in layout.branches(group) if o.fullPathName.endswith('+'+pcb_path)]
        check(len(matches)==1,'Ambiguous registered carrier')
        return matches[0].bRepBodies.itemByName('Body1')
    expected=base['registered_pcb_bounds_mm']
    h.require_bounds(layout,app,carrier(source),expected)
    group=h.component(root,BOARD,pose)
    check(app.importManager.importToTarget(app.importManager.createSTEPImportOptions(str(step)),group.component),
          'Verified board STEP import failed')
    r['pcb_bounds_mm']=h.require_bounds(layout,app,carrier(group),expected)
    # Only consult detailed board BReps for a tool or new part entering the
    # verified board envelope. Index the prior report, not the live topology.
    return group


def world(body):
    manager=f.TemporaryBRepManager.get()
    copy=manager.copy(body.nativeObject)
    check(manager.transform(copy,body.assemblyContext.transform2),'Failed to transform '+body.name)
    return copy


def shifted(bounds):
    return [[v+d for v,d in zip(p,(97,45,6))] for p in bounds]


def key(b):
    return b.assemblyContext.fullPathName+'/'+b.name


def assess(app,h,layout,root,g,rr,rec,base,v3,r):
    round_bodies={occ.component.name:occ.bRepBodies.item(0) for occ in g['round'].childOccurrences}
    a=layout.find(g['adapter'],'Adapter_V1').bRepBodies.item(0)
    p=layout.find(g['plate'],'Mounting_Plate_V3').bRepBodies.item(0)
    h.require_bounds(layout,app,p,v3['bounds_mm'])
    check(abs(p.volume*1000-v3['volume_mm3'])<.1,'Plate differs from inspected V3')
    proxy_ref=rr['solids'][REFERENCE]
    h.require_bounds(layout,app,a,shifted(proxy_ref['bounds_mm']))
    check(abs(a.volume*1000-proxy_ref['volume_mm3'])<.1,'Retained adapter differs from reference')
    for name,b in round_bodies.items():
        data=rr['solids'][name]
        h.require_bounds(layout,app,b,shifted(data['bounds_mm']))
        check(abs(b.volume*1000-data['volume_mm3'])<.1,'Round solid differs: '+name)
    r['round_placement']=dict(local_to_assembly_matrix_cm=g['round'].transform2.asArray(),
        adapter_center_xy_mm=[97,45],adapter_bottom_top_mm=[6,53],
        beam_axis='Y',beam_diameter_mm=rr['parameters_mm']['rcBeamD'],
        beam_axis_point_mm=[97,45,6+rr['parameters_mm']['rcCentre']])
    hardware=list(g['hardware'].bRepBodies)
    equipment=[p,*layout.solids(g['camera']),*layout.solids(g['case']),*hardware]
    check(len(layout.solids(g['camera']))==6 and len(layout.solids(g['case']))==3 and
          len(round_bodies)==25 and len(hardware)==30,'Incomplete or duplicate solids')
    r['inventory']=dict(round_parts=sorted(round_bodies),retained_adapter_count=1,plate_v3_count=1,
        case_solids=3,camera_solids=6,retained_fastener_solids=len(hardware),
        board_basis='Hash-matched Recovery STEP, source pose and measured PCB bounds')
    bores=r['interface_bores_xy_d_z_mm']=[]
    for y in (20,70):
        bores.extend([h.verify_bore(p,97,y,5.6,1.5,6),h.verify_bore(a,97,y,4.5,6,12)])
    for i,s in enumerate(rr['M4_stack'],1):
        for body,station in [(a,s['adapter_pocket']),(round_bodies[LOWER],s['clamp_bore']),
                             (round_bodies[LOWER],s['counterbore'])]:
            x,y,d,lo,hi=station
            bores.append(h.verify_bore(body,x+97,y+45,d,lo+6,hi+6))
        x=s['clamp_bore'][0]+97
        bores.extend([h.verify_bore(round_bodies[f'M4_insert_{i}'],x,45,4,43,53),
                      h.verify_bore(round_bodies[f'M4_screw_{i}'],x,45,4,45,59)])
    for i,(up,low) in enumerate(rr['M8_axes_xy_d_z_mm'],1):
        for body,station in [(round_bodies[UPPER],up),(round_bodies[LOWER],low)]:
            x,y,d,lo,hi=station
            bores.append(h.verify_bore(body,x+97,y+45,d,lo+6,hi+6))
        bolt=round_bodies[f'M8_bolt_{i}']
        seat=rr['parameters_mm']['rcBoltSeat']+6
        bores.append(h.verify_bore(bolt,x+97,y+45,8,seat-50,seat))
        for name,d in [(f'M8_washer_top_{i}',8.4),(f'M8_washer_bottom_{i}',8.4),(f'M8_nut_{i}',8.1)]:
            bb=shifted(rr['solids'][name]['bounds_mm'])
            bores.append(h.verify_bore(round_bodies[name],x+97,y+45,d,bb[0][2],bb[1][2]))
    mgr=f.TemporaryBRepManager.get()
    cache={}
    def solid(b):
        k=key(b)
        if k not in cache:
            cache[k]=world(b)
        return cache[k]
    trial,=[t for t in rec['placements']['equipment_envelope_midpoint']['trials']
            if abs(t['adapter_height_mm']-47)<TOL and t['beam_height_mm']==50]
    bearing=r['bearing_area_mm2']={}
    for label,mate,z,expected in [('plate',p,6,trial['mating_area_against_undrilled_faces_mm2']['base']),
                                  ('round_lower',round_bodies[LOWER],53,rr['adapter_bearing_mm2']['available'])]:
        imprint=mgr.imprintOverlapBodies(solid(a),solid(mate),False)
        check(imprint[0],'Cannot measure '+label+' seat')
        area=sum(face.area*100 for face in imprint[3] if c.Plane.cast(face.geometry) and
                 abs(face.geometry.normal.z)>.99999 and abs(face.pointOnFace.z*10-z)<TOL)
        # The two new blind V3 seats remove annuli beyond the adapter clearance holes.
        if label=='plate':
            expected-=2*math.pi*((5.6/2)**2-(4.5/2)**2)
        bearing[label]=dict(z_mm=z,measured=area,expected=expected)
        check(abs(area-expected)<.1,'Unexpected contact area at '+label)

    camera=layout.find(g['camera'],layout.CAMERA)
    usb=camera.bRepBodies.itemByName('USB cable')
    lens=camera.component.bRepBodies.itemByName('Lens')
    ends=sorted([(face.pointOnFace.y,face.createForAssemblyContext(camera).centroid)
                 for face in lens.faces if c.Plane.cast(face.geometry) and abs(face.geometry.normal.y)>.99999],
                key=lambda item:item[0])
    rear,front=[q for _,q in ends]
    angle=math.degrees(rear.vectorTo(front).angleTo(layout.V(0,0,-1)))
    front_mm=[v*10 for v in front.asArray()]
    check(angle<=.01 and math.dist(front_mm,base['lens']['front_mm'])<TOL,'Camera pose mismatch')
    r['lens']=dict(degrees_from_down=angle,front_mm=front_mm)
    shell=layout.find(g['case'],layout.CASE)
    bracket=layout.find(g['camera'],layout.BRACKET)
    v5=h.read('jetson-case-v5-report.json')['mounting_plate_interface']
    mate_axes=[layout.bore_axis(shell,v['center_pcb_xy_mm'],(0,1),4.5,v5['plate_contact_z_mm'],2)
               for v in v5['holes']]
    for slot in h.read('camera-mount-v3-report.json')['mounting_plate_slots']:
        ends=[layout.bore_axis(bracket,xy,(0,2),4.5,0,1) for xy in slot['end_arc_centers_xz_mm']]
        mate_axes.append([sum(coords)/2 for coords in zip(*ends)])
    plate_axes=[h.verify_bore(p,x,y,4.5,0,6) for x,y in v3['original_through_bores_xy_mm']]
    errors=[math.dist(m[:2],q[:2]) for m,q in zip(mate_axes,plate_axes)]
    check(len(errors)==6 and max(errors)<=TOL,'Retained case or camera shifted relative to plate')
    r['retained_mount_axis_errors_mm']=errors
    added=list(round_bodies.values())
    distances={name:app.measureManager.measureMinimumDistance(usb,b).value*10 for name,b in round_bodies.items()}
    check(all(math.isfinite(v) and v>=0 for v in distances.values()),'Invalid USB distance')
    nearest=min(distances,key=distances.get)
    r['usb_stub']=dict(nearest_round_part=nearest,distance_mm=distances[nearest],
                       lower_bound_over_slot_travel_mm=max(0,distances[nearest]-4))

    boxes={key(b):layout.box(app,b) for b in [a,*added,*equipment]}
    board_box=rec['registration']['board_bounds_plate_frame_mm']
    board=g['board']
    pcb_path=h.read('fusion-inspection.json')['pcb_occurrence']
    carrier,=[o for o in layout.branches(board) if o.fullPathName.endswith('+'+pcb_path)]
    prefix=carrier.fullPathName[:-len(pcb_path)]
    records=rec['fresh_step_inventory']['bodies']
    to_root=h.helper('CreateAdapterV1').root_box_from_local
    board_index={}
    contexts={}
    def board_candidates(bb):
        if not layout.overlaps(bb,board_box):
            return []
        if not board_index:
            board_index.update({k:to_root(row['bounds_step_mm'],board.transform2.asArray()) for k,row in records.items()})
            contexts.update({o.fullPathName:o for o in layout.branches(board)})
        found=[]
        for path,expected in board_index.items():
            if layout.overlaps(bb,expected):
                node,name=path.rsplit('/',1)
                b=contexts[prefix+node].bRepBodies.itemByName(name)
                if key(b) not in boxes:
                    boxes[key(b)]=h.require_bounds(layout,app,b,expected)
                found.append(b)
        return found
    def hits(temp,bb,targets):
        result=[]
        for b in targets:
            if layout.overlaps(bb,boxes[key(b)]):
                overlap=h.boolean(mgr.copy(temp),solid(b),f.BooleanTypes.IntersectionBooleanType)
                volume=overlap.volume*1000 if overlap.lumps.count else 0.
                if volume>.001:
                    result.append(dict(part=key(b),volume_mm3=volume))
        return result
    r['changed_interface_hits']={}
    for name,b in round_bodies.items():
        bb=boxes[key(b)]
        found=hits(solid(b),bb,[a,*equipment,*board_candidates(bb)])
        if found:
            r['changed_interface_hits'][name]=found

    # Use full axial insertion corridors, not just the final seated tool.
    all_boxes=[*boxes.values(),board_box]
    low=min(bb[0][2] for bb in all_boxes)-40
    high=max(bb[1][2] for bb in all_boxes)+40
    tools=r['tool_access']=[]
    def probe(label,stage,x,y,z0,z1,d,targets,include_board,required):
        rad=d/2
        bb=[[x-rad,y-rad,z0],[x+rad,y+rad,z1]]
        tested=targets+(board_candidates(bb) if include_board else [])
        result=hits(h.cylinder(x,y,z0,z1,d),bb,tested)
        tools.append(dict(tool=label,stage=stage,xy_mm=[x,y],z_range_mm=[z0,z1],diameter_mm=d,
                          required_for_sequence=required,hits=result))
        return result
    for path in rr['tool_paths']:
        label=path['tool']
        x,y=[v+t for v,t in zip(path['xy_mm'],(97,45))]
        z0,z1=[v+6 for v in path['z_mm']]
        if label=='M4 key before liner and beam':
            i=1 if x<97 else 2
            # Own screw is the driving target; the beam/liners close this entrance.
            targets=[b for name,b in round_bodies.items() if name!=f'M4_screw_{i}']+[a]
            probe(label,'lower_and_adapter_before_liners',x,y,z0,high,3,[round_bodies[LOWER],a],False,True)
            probe(label,'closed_clamp_before_plate',x,y,z0,high,3,targets,False,False)
            probe(label,'after_plate',x,y,z0,high,3,targets+equipment,True,False)
        else:
            i=(1 if x<97 else 3)+(y>45)
            targets=[b for name,b in round_bodies.items() if name not in {f'M8_bolt_{i}',f'M8_nut_{i}'}]+[a]
            if label=='M8 head socket':
                z1=high
            else:
                z0=low
            probe(label,'before_plate',x,y,z0,z1,20,targets,False,True)
            probe(label,'after_plate',x,y,z0,z1,20,targets+equipment,True,False)
    routes=r['flange_driver_routes']=[]
    for y in (20,70):
        for stage,targets,installed in [('before_plate',added+[a],False),('after_plate',added+[a]+equipment,True)]:
            seated=probe('Plate flange M4 driver',stage,97,y,17,47,10,targets,installed,True)
            overhead=probe('Plate flange M4 overhead approach',stage,97,y,17,high,10,targets,installed,False)
            # A compact 30 mm driver can also enter sideways under the beam.
            # Sweep a conservative rectangular envelope from outside the assembly.
            start=(min(b[0][1] for b in all_boxes)-10 if y<45 else max(b[1][1] for b in all_boxes)+10)
            ymin,ymax=min(start,y)-5,max(start,y)+5
            bb=[[92,ymin,17],[102,ymax,47]]
            prism=mgr.createBox(c.OrientedBoundingBox3D.create(h.point(97,(ymin+ymax)/2,32),
                layout.V(1,0,0),layout.V(0,1,0),1,(ymax-ymin)/10,3))
            side_hits=hits(prism,bb,targets+(board_candidates(bb) if installed else []))
            routes.append(dict(xy_mm=[97,y],stage=stage,side_sweep_bounds_mm=bb,side_hits=side_hits,
                               overhead_hits=overhead,accessible=not seated and (not overhead or not side_hits)))
    r['sequence']=['Seat M4 inserts. Fit lower clamp to adapter using both M4x14 screws before liners or beam.',
        'Fit liners, beam and upper clamp. Fit and tighten four M8x50 bolts before attaching equipment.',
        'Bring V3 equipment plate to adapter and drive both M4x10 screws from above the flange; use the measured clear driver route for each screw.',
        'For service remove plate first for blocked M8 approaches; remove upper clamp, beam and liner before accessing the two M4x14 screws.']
    r['scope']='Changed interfaces and straight tool approaches. Unchanged internal contacts inherited from source reports. USB is the modelled stub; actual cable bend and handle swing unmeasured.'
    r['blocked_after_plate']=[dict(tool=t['tool'],xy_mm=t['xy_mm']) for t in tools
                              if t['stage']=='after_plate' and t['hits']]
    r['status']=('geometry_review_required' if r['changed_interface_hits'] or distances[nearest]<=TOL
                 or any(t['required_for_sequence'] and t['hits'] for t in tools)
                 or any(not route['accessible'] for route in routes) else 'cad_prototype_review_required')


def build(app,r):
    h=full_helpers()
    layout=h.helper('CreateMountingLayoutV1')
    base,rr,rec,v3,step=reference_data(h,r)
    src=matching_root(app,FULL)
    clamp=matching_root(app,ROUND)
    names_match(clamp.occurrences,rr['solids'])
    doc=app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
    doc.name=TITLE
    design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
    if design.designIntent!=f.DesignIntentTypes.HybridDesignIntentType:
        design.designIntent=f.DesignIntentTypes.HybridDesignIntentType
    design.designType=f.DesignTypes.DirectDesignType
    root=design.rootComponent
    groups=retained(h,layout,root,src,r)
    groups['board']=registered_board(app,h,layout,src,root,rec,base,step,r)
    group=groups['round']=h.component(root,'Round beam clamp assembly',h.shift(97,45,6))
    for occ in clamp.occurrences:
        same_pose(occ.transform2,h.identity())
        check(occ.bRepBodies.count==1 and not occ.childOccurrences.count,'Unexpected round clamp branch: '+occ.name)
        if occ.component.name!=REFERENCE:
            copy_tree(h,group.component,occ)
    names_match(root.occurrences,[v.component.name for v in groups.values()])
    assess(app,h,layout,root,groups,rr,rec,base,v3,r)
    view=app.activeViewport
    cam=view.camera
    cam.viewOrientation=c.ViewOrientations.IsoTopRightViewOrientation
    cam.isFitView=True
    view.camera=cam
    view.fit()


def run(context):
    app=c.Application.get()
    r=dict(status='started',document=TITLE,units='mm',approved_for_print=False,
           approved_for_installation=False,strength_tested=False)
    try:
        build(app,r)
    except Exception:
        r.update(status='incomplete',error=traceback.format_exc())
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=DOCS,delete=False) as out:
        json.dump(r,out,indent=2,allow_nan=False)
    os.replace(out.name,REPORT)
    app.userInterface.messageBox(r['status']+'\n'+str(REPORT))
