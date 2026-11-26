"""Read-only geometry diagnostics; all Boolean targets are disposable temporary stock."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

import adsk
import adsk.core as c
import adsk.fusion as f

ROOT = Path(__file__).resolve().parents[2]
STEP = ROOT/'Reference'/'Jetson_Nano_Dev_Kit_3D_b01.stp'
REPORT = ROOT/'Documentation'/'jetson-reference-diagnostics-report.json'
ASSEMBLY = '682-13450_PORG_ME_THER_ASSY:1'
TARGETS = [f'{ASSEMBLY}+155-0717-000:{i}' for i in range(1,5)]
FRAMES = ('native','assembly_proxy','temporary_before','temporary_after')
HIGH = f.CalculationAccuracy.VeryHighCalculationAccuracy
DIFF = f.BooleanTypes.DifferenceBooleanType
COMMON = f.BooleanTypes.IntersectionBooleanType
XYZ_TOL = .01
VOL_REL = VOL_ABS = 1e-7  # Same test as RecoverJetsonReferenceV1; units mm3 for absolute tolerance.


def save(report):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=REPORT.parent,
                                     suffix='.tmp', delete=False) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    os.replace(stream.name, REPORT)


def finite(value):
    if not math.isfinite(value):
        raise ValueError('Non-finite measurement')
    return value


def bounds(box):
    return [[finite(10*v) for v in p.asArray()] for p in (box.minPoint,box.maxPoint)]


def tight_bounds(app,body):
    box=app.measureManager.getOrientedBoundingBox(
        body,c.Vector3D.create(1,0,0),c.Vector3D.create(0,1,0))
    centre=[10*v for v in box.centerPoint.asArray()]
    half=[5*v for v in (box.length,box.width,box.height)]
    return [[finite(a+s*b) for a,b in zip(centre,half)] for s in (-1,1)]


def read_fields(operations):
    row={}
    for key,op in operations.items():
        try:
            row[key]=op()
        except Exception as exc:
            row[key]=None
            row.setdefault('errors',{})[key]=str(exc)
    return row


def metrics(app,body):
    return read_fields({
        'isValid':lambda:body.isValid,'isSolid':lambda:body.isSolid,
        'faces':lambda:body.faces.count,'edges':lambda:body.edges.count,'vertices':lambda:body.vertices.count,
        'volume_mm3':lambda:finite(body.volume*1000),
        'very_high_volume_mm3':lambda:finite(body.getPhysicalProperties(HIGH).volume*1000),
        'bounding_box_mm':lambda:bounds(body.boundingBox),
        'precise_bounding_box_mm':lambda:bounds(body.preciseBoundingBox),
        'axis_aligned_measurement_mm':lambda:tight_bounds(app,body)})


def stock(manager,box):
    # Oversize on all six sides so the stock introduces no coincident boundary.
    lo,hi=box
    centre=c.Point3D.create(*[(a+b)/20 for a,b in zip(lo,hi)])
    dims=[(b-a+10)/10 for a,b in zip(lo,hi)]
    return manager.createBox(c.OrientedBoundingBox3D.create(
        centre,c.Vector3D.create(1,0,0),c.Vector3D.create(0,1,0),*dims))


def persist_measurement(app, root, body, label, record):
    raw = record['temporary'] = metrics(app, body)
    raw_lumps = body.lumps.count
    stored = root.bRepBodies.add(body)
    if not stored or stored.isTemporary:
        raise ValueError('Could not materialize ' + label)
    stored.name = label
    measured = record['persisted'] = metrics(app, stored)
    record['body_name'] = stored.name
    for state in (raw, measured):
        if (state.get('errors') or not state['isValid'] or not state['isSolid']
                or min(state[k] for k in ('faces','edges','vertices','volume_mm3')) <= 0):
            raise ValueError('Incomplete solid measurement: ' + label)
        state['very_high_minus_default_mm3'] = state['very_high_volume_mm3'] - state['volume_mm3']
    record['precise_box_errors_mm'] = [
        math.dist(p,q) for p,q in zip(raw['precise_bounding_box_mm'], measured['precise_bounding_box_mm'])]
    record['topology_delta'] = {k: measured[k]-raw[k] for k in ('faces','edges','vertices')}
    if (stored.lumps.count != raw_lumps or any(record['topology_delta'].values())
            or max(record['precise_box_errors_mm']) > XYZ_TOL or measured['very_high_volume_mm3'] <= 0):
        raise ValueError('Materialization requires review: ' + label)
    return measured


def region(app,root,label,manager,box,operations,record):
    target=stock(manager,box)
    for op,tool in operations:
        if not manager.booleanOperation(target,tool,op):
            raise ValueError('Kernel Boolean operation returned false')
    if not target.isValid:
        raise ValueError('Invalid Boolean result; empty volume unconfirmed')
    lumps=target.lumps.count
    record.update(lumps=lumps, faces=target.faces.count, edges=target.edges.count,
                  vertices=target.vertices.count)
    if not any(record[k] for k in ('lumps','faces','edges','vertices')):
        record.update(kind='empty', volume_mm3=0.)
        return
    if not lumps or not target.isSolid:
        raise ValueError('Boolean result contains non-solid geometry')
    record['kind'] = 'solid'
    measured = persist_measurement(app,root,target,label,record)
    record['volume_mm3'] = measured['very_high_volume_mm3']
    return target


def close_volume(a,b):
    return math.isclose(a,b,rel_tol=VOL_REL,abs_tol=VOL_ABS)


def compare(app,root,label,a,b,ma,mb,assembly_solids=None):
    # Use the actual input bodies as read-only tools, even for A minus B.
    # Copying A first would itself introduce the transformation under investigation.
    manager=f.TemporaryBRepManager.get()
    ba,bb=ma['axis_aligned_measurement_mm'],mb['axis_aligned_measurement_mm']
    box=[[min(p,q) for p,q in zip(ba[0],bb[0])],[max(p,q) for p,q in zip(ba[1],bb[1])]]
    result={}
    for name,ops in [('a_only',[(DIFF,b),(COMMON,a)]), ('b_only',[(DIFF,a),(COMMON,b)]),
                     ('overlap',[(COMMON,a),(COMMON,b)])]:
        record=result[name]={}
        try:
            solid=region(app,root,label+'_'+name,manager,box,ops,record)
            if assembly_solids is not None and solid is not None:
                assembly_solids[name]=solid
        except Exception as exc:
            result.setdefault('errors',{})[name]=str(exc)
    result['bbox_corner_errors_mm']=[math.dist(p,q) for p,q in zip(ba,bb)]
    result['topology_delta_b_minus_a']={k:mb[k]-ma[k] for k in ('faces','edges','vertices')}
    for key in ('volume_mm3','very_high_volume_mm3'):
        result['delta_'+key]=mb[key]-ma[key]
    if result.get('errors'):
        result['classification']='inconclusive'
        return result
    left,right,intersection=[result[k]['volume_mm3'] for k in ('a_only','b_only','overlap')]
    va,vb=ma['very_high_volume_mm3'],mb['very_high_volume_mm3']
    result['symmetric_difference_mm3']=left+right
    closure=close_volume(left+intersection,va) and close_volume(right+intersection,vb)
    result['volume_balance_residuals_mm3']=[left+intersection-va,right+intersection-vb]
    result['volume_balance_ok']=closure
    if not closure:
        outcome='inconclusive'
    elif left+right>max(VOL_ABS,VOL_REL*max(va,vb)):
        outcome='material_difference_in_compared_frame'
    elif (result['a_only']['lumps'] or result['b_only']['lumps']
          or intersection<=0 or not close_volume(va,vb) or max(result['bbox_corner_errors_mm'])>XYZ_TOL):
        outcome='inconclusive'
    else:
        outcome='topology_only' if any(result['topology_delta_b_minus_a'].values()) else 'equivalent'
    result['classification']=outcome
    return result


def verify_assembly_results(app,root,prefix,manager,inverse,solids,row):
    comparison=row['comparisons']['proxy_vs_after_root']
    native=row['states']['native']
    proxy=row['states']['assembly_proxy']
    after=row['materializations']['proxy_vs_after_root']['persisted']
    roundtrip=row['materializations']['native_vs_roundtrip_local']['persisted']
    evidence=row['assembly_boolean_in_native_frame']={'regions':{},'passed':False}
    if comparison.get('errors') or 'volume_balance_ok' not in comparison:
        raise ValueError('Assembly Boolean measurements incomplete')
    volumes={}
    for name in ('a_only','b_only','overlap'):
        source=comparison[name]
        record=evidence['regions'][name]={'assembly_kind':source['kind']}
        if source['kind']=='empty':
            # A rigid transform preserves the empty set. No solid exists to copy.
            volumes[name]=record['local_volume_mm3']=0.
            continue
        local=manager.copy(solids[name])
        if not local or not manager.transform(local,inverse):
            raise ValueError('Cannot transform Assembly Boolean result: '+name)
        measured=persist_measurement(app,root,local,prefix+'_local_'+name,record)
        volumes[name]=record['local_volume_mm3']=measured['very_high_volume_mm3']
    overlap=volumes['overlap']
    va=native['very_high_volume_mm3']
    vb=roundtrip['very_high_volume_mm3']
    totals=[volumes['a_only']+overlap,volumes['b_only']+overlap]
    evidence['reference_volumes_mm3']=[va,vb]
    evidence['volume_balance_residuals_mm3']=[totals[0]-va,totals[1]-vb]
    evidence['volume_balance_ok']=close_volume(totals[0],va) and close_volume(totals[1],vb)
    def box_errors(a,b):
        return [math.dist(p,q) for p,q in zip(a['precise_bounding_box_mm'],b['precise_bounding_box_mm'])]
    evidence['assembly_precise_box_errors_mm']=box_errors(proxy,after)
    if overlap<=0:
        return
    local_overlap=evidence['regions']['overlap']['persisted']
    evidence['local_overlap_precise_box_errors_mm']=box_errors(native,local_overlap)
    evidence['passed']=all((
        comparison['a_only']['kind']=='empty', comparison['b_only']['kind']=='empty',
        close_volume(comparison['overlap']['volume_mm3'],after['very_high_volume_mm3']),
        evidence['volume_balance_ok'], close_volume(va,vb),
        max(comparison['bbox_corner_errors_mm'])<=XYZ_TOL,
        max(evidence['assembly_precise_box_errors_mm'])<=XYZ_TOL,
        max(evidence['local_overlap_precise_box_errors_mm'])<=XYZ_TOL))


def inspect(app,root,occ,row,report,checkpoint=save):
    proxy=occ.bRepBodies.itemByName('Body1')
    if not proxy or not proxy.assemblyContext or proxy.assemblyContext.fullPathName!=occ.fullPathName:
        raise ValueError('Body1 root-context proxy not uniquely identified')
    native=proxy.nativeObject
    pose=occ.transform2
    identity=c.Matrix3D.create().asArray()
    row['occurrence_chain_leaf_first']=[]
    node=occ
    while node:
        direct=node.nativeObject or node
        row['occurrence_chain_leaf_first'].append(dict(path=node.fullPathName,
            local_to_parent=direct.transform2.asArray(),contextual_to_root=node.transform2.asArray()))
        node=node.assemblyContext
    states=row['states']={}
    manager=f.TemporaryBRepManager.get()
    for key,body in [('native',native),('assembly_proxy',proxy)]:
        states[key]=metrics(app,body)
    # Measure the SAME copy before/after transform, reproducing the recovery code.
    tmp=manager.copy(native)
    states['temporary_before']=metrics(app,tmp)
    row['matrices_local_to_state_cm']={k:(identity if k in ('native','temporary_before')
                                         else pose.asArray()) for k in FRAMES}
    checkpoint(report)
    results=row['comparisons']={}
    records=row['materializations']={}
    prefix=occ.name.replace(':','_')
    assembly_solids={}
    def test(key,a,b,ma,retain=None):
        try:
            record=records[key]={}
            mb=persist_measurement(app,root,b,prefix+'_'+key,record)
            results[key]=compare(app,root,prefix+'_'+key,a,b,ma,mb,retain)
        except Exception as exc:
            results[key]=dict(classification='inconclusive',error=str(exc))
    test('native_vs_before_local',native,tmp,states['native'])
    if not manager.transform(tmp,pose):
        raise ValueError('Transform of temporary native copy failed')
    states['temporary_after']=metrics(app,tmp)
    checkpoint(report)
    test('proxy_vs_after_root',proxy,tmp,states['assembly_proxy'],assembly_solids)
    # Independent pose check using corresponding source vertices (no index mapping on copy).
    checks=row['native_to_proxy_vertices_mm']=[]
    for vertex in native.vertices:
        mapped=vertex.geometry.copy()
        if not mapped.transformBy(pose):
            raise ValueError('Native vertex transform failed')
        measured=vertex.createForAssemblyContext(occ).geometry
        checks.append(dict(expected=[10*v for v in mapped.asArray()],proxy=[10*v for v in measured.asArray()],
                           error_mm=finite(10*mapped.distanceTo(measured))))
    # A second common-frame comparison avoids deriving equivalence solely from
    # a Boolean operation involving an assembly proxy.
    inverse=pose.copy()
    if not inverse.invert():
        raise ValueError('Non-invertible occurrence transform')
    row['root_to_native_matrix_cm']=inverse.asArray()
    back=manager.copy(tmp)
    if not manager.transform(back,inverse):
        raise ValueError('Inverse transform of disposable copy failed')
    test('native_vs_roundtrip_local',native,back,states['native'])
    verify_assembly_results(app,root,prefix,manager,inverse,assembly_solids,row)
    row['changes_from_native']={}
    for key,state in states.items():
        row['changes_from_native'][key]=read_fields({
            k:lambda k=k:state[k]-states['native'][k]
            for k in ('volume_mm3','very_high_volume_mm3','faces','edges','vertices')})
    checkpoint(report)
    if (any(s.get('errors') or not s['isSolid'] or not s['isValid'] or s['volume_mm3']<=0
            for s in states.values()) or not checks
            or not close_volume(states['native']['very_high_volume_mm3'],
                                states['assembly_proxy']['very_high_volume_mm3'])):
        row['status']='inconclusive'
    elif max(p['error_mm'] for p in checks)>XYZ_TOL:
        row['status']='frame_mismatch'
    elif any(r['classification']=='material_difference_in_compared_frame' for r in results.values()):
        row['status']='material_difference_in_compared_frame'
    elif (row['assembly_boolean_in_native_frame']['passed'] and
          all(results[key]['classification'] in ('equivalent','topology_only')
              for key in ('native_vs_before_local','native_vs_roundtrip_local'))):
        row['status']='geometry_equivalent'
    else:
        row['status']='inconclusive'
    # Recovery consumes this exact STEP-root solid after checking the verdict.
    return tmp


def run(context):
    app=c.Application.get()
    report=dict(status='importing',step_path=str(STEP),length_unit='mm',volume_unit='mm3',
        matrix_format='row-major 4x4; translations cm',occurrences={},xyz_tol_mm=XYZ_TOL,
        volume_relative_tolerance=VOL_REL,volume_absolute_tolerance_mm3=VOL_ABS)
    doc=None
    try:
        with STEP.open('rb') as stream:
            report['step_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
        if report['step_sha256']!='054c594a8b08185f8aab22e30976f7d7df3f16771455a1485381111e068f279d':
            raise ValueError('STEP differs from the inspected project reference')
        manager=app.importManager
        doc=manager.importToNewDocument(manager.createSTEPImportOptions(str(STEP)))
        if not doc or not doc.isValid:
            raise ValueError('STEP import failed')
        doc.activate()
        adsk.doEvents()
        design=f.Design.cast(doc.products.itemByProductType('DesignProductType'))
        if not design:
            raise ValueError('Imported document has no Design product')
        occurrences={}
        for occ in design.rootComponent.allOccurrences:
            if occ.fullPathName in TARGETS:
                occurrences.setdefault(occ.fullPathName,[]).append(occ)
        scratch=app.documents.add(c.DocumentTypes.FusionDesignDocumentType)
        diagnostic=f.Design.cast(scratch.products.itemByProductType('DesignProductType'))
        diagnostic.designType=f.DesignTypes.DirectDesignType
        report['diagnostic_document']=scratch.name
        report['measurement_method']='STEP root proxy Booleans; actual Boolean solids inverse-transformed to native coordinates and persisted for VeryHighCalculationAccuracy volume balance. Assembly and local results reported separately.'
        for path in TARGETS:
            row=report['occurrences'][path+'/Body1']={}
            try:
                matches=occurrences.get(path,[])
                if len(matches)!=1:
                    raise ValueError(f'Expected exactly one occurrence; found {len(matches)}')
                inspect(app,diagnostic.rootComponent,matches[0],row,report)
            except Exception:
                row.update(status='inconclusive',error=traceback.format_exc())
            save(report)
        equivalent=all(row['status']=='geometry_equivalent' for row in report['occurrences'].values())
        report['status']='all_four_geometry_equivalent' if equivalent else 'review_required'
        report['recovery_recommendation']=(
            'Keep all 923 bodies and existing gates. For a count/default-volume mismatch, accept a body only after '
            'materializing temporary inputs and nonempty Boolean results in a separate direct design, confirming '
            'same-frame bidirectional empty material differences, positive overlap, very-high-accuracy '
            'volume balance and placement checks all succeed for that occurrence.'
            if equivalent else 'Keep recovery blocked. Review individual Boolean results, non-overlap volumes, '
            'topology changes and frame residuals before changing the recovery method.')
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    finally:
        report['imported_document_left_open']=bool(doc and doc.isValid)
        if report['imported_document_left_open']:
            report['document_name']=doc.name
        save(report)
        if report['imported_document_left_open']:
            doc.activate()
    app.userInterface.messageBox(report['status']+'\n'+str(REPORT))
