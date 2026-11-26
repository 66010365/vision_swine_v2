import datetime
import json
import time
import adsk
import os
import traceback
from pathlib import Path

import adsk.core as core
import adsk.fusion as fusion


PROJECT = Path(__file__).resolve().parents[2]


def run(context):
    app = core.Application.get()
    ui = app.userInterface
    folder = PROJECT / 'Documentation'
    folder.mkdir(exist_ok=True)
    log_path = folder / 'fusion-inspection.log'
    output = folder / 'fusion-inspection.json'
    result = dict(status='starting', units='mm', mounting_holes=None, ports=[], ports_complete=False)

    def log(message):
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(f'{datetime.datetime.now().isoformat()} {message}\n')

    def save():
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temp, output)

    def pick(prompt, filters):
        if app.activeDocument != doc:
            raise ValueError('Imported document is no longer active')
        try:
            selection = ui.selectEntity(prompt, filters)
        except RuntimeError as exc:
            if 'selections.size()' not in str(exc) and 'cancel' not in str(exc).lower():
                raise
            selection = None
        if not selection or not selection.entity:
            raise InterruptedError('No selection. Partial measurements saved.')
        return selection.entity

    try:
        log('Starting STEP inspection')
        source = PROJECT / 'Reference' / 'Jetson_Nano_Dev_Kit_3D_b01.stp'
        result.update(source=source.name, status='importing')
        save()
        if not source.is_file():
            raise FileNotFoundError(source)
        manager = app.importManager
        options = manager.createSTEPImportOptions(str(source))
        doc = manager.importToNewDocument(options) if options else None
        if not doc or not doc.isValid:
            raise ValueError('STEP import did not return a valid document')
        log(f'Imported {doc.name}; activating document')
        doc.activate()
        deadline = time.monotonic() + 5
        adsk.doEvents()
        while app.activeDocument != doc and time.monotonic() < deadline:
            time.sleep(0.05)
            adsk.doEvents()
        if app.activeDocument != doc:
            raise ValueError('Fusion did not activate the imported document')
        design = fusion.Design.cast(doc.products.itemByProductType('DesignProductType'))
        if not design:
            raise ValueError('Imported document has no Design product')
        root = design.rootComponent
        app.activeViewport.fit()
        app.activeViewport.refresh()
        adsk.doEvents()
        solids = [b for b in root.bRepBodies if b.isSolid]
        for occ in root.allOccurrences:
            solids.extend(b for b in occ.bRepBodies if b.isSolid)
        log(f'Active document {doc.name}; root={root.name}; solids={len(solids)}')
        reference = '142-13449-1000-B01-SKU0000'
        candidates = []
        for body in solids:
            if reference not in (body.parentComponent.name + ' ' + body.name).upper():
                continue
            planes = [f for f in body.faces if core.Plane.cast(f.geometry)]
            if len(planes) < 2:
                continue
            face = max(planes, key=lambda f: f.area)
            lines = [e for e in face.edges if core.Line3D.cast(e.geometry)]
            if not lines:
                continue
            edge = max(lines, key=lambda e: e.length)
            x = edge.startVertex.geometry.vectorTo(edge.endVertex.geometry)
            x.normalize()
            z = face.geometry.normal
            y = z.crossProduct(x)
            y.normalize()
            box = app.measureManager.getOrientedBoundingBox(body, x, y)
            dims = [10*v for v in (box.length, box.width, box.height)]
            log(f'PCB candidate {body.name} measured axes (mm)={dims}')
            # Match measured STEP vertex spans; these tolerances classify, never supply dimensions.
            if all(abs(a-b)<0.2 for a,b in zip(dims,(100,79,1.57))):
                candidates.append((body,x,z,face.pointOnFace))
        if len(candidates) != 1:
            raise ValueError(f'Expected one matching PCB solid, found {len(candidates)}')
        pcb,x,z,origin = candidates[0]
        # All geometry here is already expressed in root coordinates by the occurrence proxies.
        sink = [b for b in solids if b.assemblyContext and
                'PORG_HSK_ASSY' in b.assemblyContext.fullPathName.upper()]
        if not sink:
            raise ValueError('Cannot establish PCB top side: PORG_HSK_ASSY missing')
        y = z.crossProduct(x)
        offsets = [origin.vectorTo(app.measureManager.getOrientedBoundingBox(b,x,y).centerPoint).dotProduct(z)
                   for b in sink]
        if max(offsets) < -0.157:
            z.scaleBy(-1)
        elif min(offsets) <= 0.157:
            raise ValueError('Heatsink bodies do not establish an unambiguous top side')
        y = z.crossProduct(x)
        y.normalize()
        log(f'PCB={pcb.name}; root-context axes X={x.asArray()} Y={y.asArray()} Z={z.asArray()}')
        def coords(point):
            v = origin.vectorTo(point)
            return [10 * v.dotProduct(axis) for axis in (x, y, z)]

        def bounds(entity):
            box = app.measureManager.getOrientedBoundingBox(entity, x, y)
            c = coords(box.centerPoint)
            size = [10 * a for a in (box.length, box.width, box.height)]
            return dict(min=[a-b/2 for a,b in zip(c,size)],
                        max=[a+b/2 for a,b in zip(c,size)], size=size)

        board = bounds(pcb)
        # Anchor XY to minimum board extents and Z to the selected underside.
        origin.translateBy(scaled_vector(x, board['min'][0]/10))
        origin.translateBy(scaled_vector(y, board['min'][1]/10))
        board = bounds(pcb)
        origin.translateBy(scaled_vector(z, board['min'][2]/10))
        board = bounds(pcb)
        envelopes = [bounds(b) for b in solids]
        lowest = min(b['min'][2] for b in envelopes)
        highest = max(b['max'][2] for b in envelopes)
        result.update(status='measured', units='mm', source=source.name, pcb=board,
                      frame=dict(origin_assembly_mm=[10*v for v in origin.asArray()],
                                 x=x.asArray(), y=y.asArray(), z=z.asArray()),
                      height_including_heatsink=highest-lowest,
                      underside_projection=max(0,-lowest), mounting_holes=None,
                      ports=[], ports_complete=False)
        save()
        cylinders = {}
        for face in pcb.faces:
            cylinder = core.Cylinder.cast(face.geometry)
            if cylinder and abs(cylinder.axis.dotProduct(z)) > 0.999999:
                cx, cy = coords(cylinder.origin)[:2]
                diameter = 20 * cylinder.radius
                key = tuple(round(v, 5) for v in (cx, cy, diameter))
                cylinders[key] = dict(center_xy=[cx, cy], diameter=diameter)
        result['pcb_axial_cylinders'] = list(cylinders.values())
        result['pcb_occurrence'] = pcb.assemblyContext.fullPathName if pcb.assemblyContext else root.name
        save()
        log(f'Board and assembly measured; height={highest-lowest} mm')
        answer = ui.messageBox('Confirm four known mounting holes now? Choose No to skip to ports.',
                               'InspectJetson', core.MessageBoxButtonTypes.YesNoButtonType)
        confirm_holes = answer == core.DialogResults.DialogYes
        holes = []
        for i in range(4 if confirm_holes else 0):
            wall = pick(f'Select cylindrical wall of confirmed mounting hole {i+1}', 'CylindricalFaces')
            cyl = wall.geometry
            if wall.body != pcb or abs(cyl.axis.dotProduct(z)) < 0.999999:
                raise ValueError('Hole must be perpendicular to selected PCB')
            center = coords(cyl.origin)[:2]
            if any(sum((a-b)**2 for a,b in zip(center,h['center_xy'])) < 1e-6 for h in holes):
                raise ValueError('Duplicate hole')
            holes.append(dict(center_xy=center, diameter=20*cyl.radius))
        result['mounting_holes'] = holes or None
        save()
        while True:
            label, cancelled = ui.inputBox('Port name or designator. Enter DONE when all ports are recorded. Cancel for partial results.')
            if cancelled:
                break
            if label.strip().upper() == 'DONE':
                result['ports_complete'] = True
                save()
                break
            if not label.strip():
                continue
            occ = pick('Select complete connector occurrence in the Browser', 'Occurrences')
            prefix = occ.fullPathName
            parts = [bounds(b) for b in solids if b.assemblyContext and
                     (b.assemblyContext.fullPathName == prefix or
                      b.assemblyContext.fullPathName.startswith(prefix + '+'))]
            if not parts:
                raise ValueError('Selected connector contains no solid bodies')
            lo = [min(p['min'][i] for p in parts) for i in range(3)]
            hi = [max(p['max'][i] for p in parts) for i in range(3)]
            envelope = dict(min=lo, max=hi, size=[b-a for a,b in zip(lo,hi)])
            result['ports'].append(dict(name=label, occurrence=occ.fullPathName,
                                        housing_bounds=envelope))
            save()
        result['status'] = 'complete' if holes and result['ports_complete'] else 'partial'
        save()
        log(result['status'])
        ui.messageBox(str(output))
    except InterruptedError as exc:
        result.update(status='partial', reason=str(exc))
        save()
        log(str(exc))
        ui.messageBox(str(exc))
    except Exception:
        detail = traceback.format_exc()
        result.update(status='failed', error=detail.splitlines()[-1])
        save()
        log(detail)
        ui.messageBox(f"{result['error']}\n{log_path}")


def scaled_vector(axis, distance):
    vector = axis.copy()
    vector.scaleBy(distance)
    return vector




