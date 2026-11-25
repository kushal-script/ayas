"""Ayas handheld enclosure, generated with the Autodesk Fusion API.

A two-hand, F1-steering-wheel-style housing for the Ayas inspection device.

Layout:
- Back face (toward the user): a 5-inch display, landscape. Default: Waveshare 5inch DSI LCD (800x480);
  the official Raspberry Pi Touch Display 2 5" is also available (P['display'] = 'td2_5').
- Behind the display: the Raspberry Pi 5 bolts onto the display's own standoffs, with its Active Cooler;
  the AI HAT+ 2 sits on its 16 mm stacking header.
- Front face (toward the part):
  - the Raspberry Pi AI Camera at the centre, under a removable hood;
  - an angled (toe-in) LED pod on each side, each holding 3 white 3-LED strips plus 3 UV strips
    (3 x 5 mm LEDs each); the UV block tilts up toward the image centre.
- Bottom of the body: a bay for a USB-PD power bank, behind a removable BayDoor.
- Top: a 30 mm exhaust fan under the top vents.
- Sides: short arms to two grips.
  - Each grip has an index-finger trigger on its front face.
  - The back of each arm root (next to the display) has two thumb buttons.

How to run (inside Fusion, or through the Fusion MCP):
    g = {}
    exec(open('<repo>/hardware/enclosure/ayas_handheld_fusion.py').read(), g)
    g['build']()                      # or g['build'](overrides={'toe_deg': 16}, export=False)

Each run creates a NEW design document. Open documents are never modified.
All dimensions are in mm.

Axes:
- X: width. +X is the right-hand side when you look at the FRONT, so it is the user's LEFT hand.
- Y: up.
- Z: depth. +Z is the FRONT (camera, lights, toward the part); -Z is the BACK (display, toward the user).

Printable bodies (component "Enclosure"):
- FrontCentre, BackLid, BayDoor, CamHood
- LedPod_L / LedPod_R
- WingFront_L / WingFront_R, WingBack_L / WingBack_R ("L"/"R" = the user's left/right hand)

Fit-check dummies (component "Reference", never printed):
display, Pi stack, camera, power bank, LEDs, fan, power-electronics carriers.
"""

import math
import os

import adsk.core
import adsk.fusion

MM = 0.1  # Fusion's internal length unit is cm

REPO_DIR = os.environ.get('AYAS_REPO', os.path.expanduser('~/Desktop/bharath_ai_soc_challenge'))   # set AYAS_REPO if the repo lives elsewhere
OUT_DIR = os.path.join(REPO_DIR, 'hardware', 'enclosure', 'out')

# ----------------------------------------------------------------------------------------------
# Parameters (mm). CHECK = from a datasheet or an estimate: measure your part with calipers.
# ----------------------------------------------------------------------------------------------
P = dict(
    wall=2.2,                     # shell wall (about 5 perimeters with a 0.4 mm nozzle)
    clear=0.4,                    # pocket clearance per side around bought parts

    # central body (hollow)
    display='ws5_dsi',            # profile from DISPLAYS below; it also sets body_w/x_split/grip_cx/pod_x_in
    body_y_bot=-66.0, body_y_top=68.0,
    body_z_back=-40.0, body_z_front=34.0,
    body_r_top=14.0, body_r_bot=5.0,   # front-view corner radii
    body_r_edge=4.0,                    # fillet on the front and back perimeter edges
    z_split_centre=-30.0,               # BackLid | FrontCentre
    door_y=-60.0,                       # BayDoor | rest of the centre piece

    # arms + grips (hollow, thinner than the body, set back toward the display)
    # The whole wing (arm + grip) is raked: its mid-plane is z = slab_zc + y*tan(grip_rake_deg), so the
    # grips lean with their bottoms toward the user (relaxed wrists at chest height). The grips also
    # splay (tops inward, like an F1 wheel) and have a palm swell (wider in the middle).
    slab_t=36.0, slab_zc=-6.0,
    arm_y_top=40.0, arm_y_bot=-40.0, arm_r_corner=6.0,
    slab_r_edge=11.0,             # front/back edge rounding: gives the grips a near-oval section
    grip_len=124.0,               # straight grip length (>= P95 hand breadth 94 mm + margin)
    grip_w_end=29.0, grip_w_mid=32.0,   # grip width at the ends / at the palm swell
    grip_r_end=13.0, grip_r_inner=12.0,   # convex corners must exceed slab_r_edge
    grip_rake_deg=12.0, grip_splay_deg=7.0,
    grip_cy=0.0, trigger_y=34.0,  # trigger height along the grip (index finger)

    disp_cy=14.0,                 # glass centre height
    disp_lip=1.2,                 # bezel lip in front of the glass edge

    # Raspberry Pi 5 + Active Cooler + AI HAT+ 2, from the official drawings/STEP. Pi-local coordinates:
    # origin at the corner of the USB-C/HDMI edge and the microSD/PCIe edge, +x toward USB/Ethernet,
    # +y toward the GPIO header, z = 0 on the Pi PCB top.
    pi_pcb_t=1.4, hat_standoff=16.0, hat_pcb_t=1.6,
    stack_top=27.0,               # heatsink push-pin heads above the Pi PCB top (est. 26.4-26.9)
    post_d=7.0, post_pad=1.0,     # display posts press through a 1 mm foam pad

    # Raspberry Pi AI Camera (official product brief). CHECK.
    cam_w=25.0, cam_h=23.862, cam_pcb_t=1.12,
    cam_lens_from_far=14.4,
    cam_holes_far=((2.0, 2.0), (23.0, 2.0), (2.0, 14.5), (23.0, 14.5)),
    cam_lens_h=8.0, cam_barrel_d=5.75, cam_set_back=1.0,
    cam_boss_d=5.5, m2_pilot=1.6,     # every service screw goes into a heat-set insert (m2_insert_d / m3_insert_d)
    cam_dfov=78.3,
    hood_l=13.0, hood_screw_dy=21.0,
    tool_hole_d=14.0,             # front-wall hole: the lens tool reaches the barrel with the hood off

    # power bank bay: Xiaomi Pocket Power Bank Pro 10000 33 W (90 x 63.9 x 24.4), + 20 mm for a
    # right-angle USB-C plug at the +X end. For the Anker Nano 10000 30 W use bay_l=106, bay_t=28.
    bay_l=92.0, bay_w=66.0, bay_t=26.5, bay_plug=20.0,
    bank=(90.0, 63.9, 24.4),

    # LED pods on the body front, either side of the camera
    pod_margin=-2.0,              # negative margin: the pod overhangs the edge fillet a little
    toe_deg=14.0,                 # atan(pod centre offset / 200 mm working distance)
    pod_base=8.0,                 # deep enough at the inner edge for the WHITE carrier pocket (7.6 mm)
    # white light: 12 V 3 x 2835 LED modules, 40 x 10 mm housing, a screw tab at each end (owned: 3609-2835-3LED)
    white_n=3, white_l=40.0, white_w=10.0, white_gap=4.0,   # 3 per pod at 14 mm pitch (the WHITE-L/R carrier)
    white_mod_h=6.0,              # module height on the carrier (VERIFY with calipers)
    white_hole_pitch=46.0,        # tab hole centres (VERIFY; hardware/pcb/design.py WHITE_MOD)
    white_led_pitch=12.5,         # LED centres along a module (photo of the owned module; VERIFY)
    white_pocket=5.0,             # hood ring: module recess depth in the ring wall
    white_pcb=(50.0, 66.0, 1.6),  # WHITE carrier (u across the face, v down from its top edge)
    white_pcb_holes=((47.0, 3.5), (47.0, 62.5)),   # 2 x M2 at the OUTER edge (pod frame u); the inner edge
                                                   # slides under a lip (the pod is thin there)
    uv_n=3, uv_per=3, uv_pitch=10.16, uv_led_d=5.3, uv_col_gap=16.0, uv_skin=3.0,
    uv_board=(30.5, 10.2),        # 12 x 4-hole perfboard per UV strip
    uv_tilt_deg=16.0,             # UV block tilted up toward the image centre
    uv_frame_h=6.0,
    # UV socket board (hardware/pcb, board 'uv'): LEDs plug in from the FRONT (flange in a 2 mm front counterbore),
    # legs pass the skin into 2-pin sockets on a 43 x 30.5 board held on two M2 bosses behind the skin
    uv_sock_h=3.0, uv_pcb=(43.0, 30.5, 1.6), uv_pcb_holes=((-8.0, -10.57), (8.0, -20.73)), uv_boss_d=5.5,

    # light-tight inspection hood (device v2, docs/twin/DESIGN_STEEL.md C2): clips over the body front,
    # rests on the part; its LED ring replaces the pods for certified inspection (app.yaml illumination: ring)
    hood_on=True,
    hood_wd=200.0,                # camera lens to plate (the twin's working distance)
    hood_bottom=(200.0, 190.0),   # outer size at the rim, centred on the optical axis
    hood_collar_z0=23.0,          # back edge of the collar that slides over the body front (clear of the arms)
    hood_rim=(5.0, 6.0),          # rim thickening (inward width, height) for the light seal
    hood_seal=(3.2, 1.6),         # groove for 3 x 3 mm black EPDM foam (width, depth)
    hood_screw=(48.0, 28.5),      # M3 thumb screws through the collar into FrontCentre inserts (y, z)
    hood_wire=(58.0, 28.5, 10.0, 5.0),   # wire slot in each collar side wall (y, z, length, height)
    ring_h=18.0, ring_inset=6.0,  # LED height above the plate; board face this far in from the wall
    ring_mods=(('y', 1, -44.0), ('y', 1, 44.0), ('y', -1, -44.0), ('y', -1, 44.0), ('x', 1, 0.0), ('x', -1, 0.0)),
    lip_len=32.0, lip_t=2.0, lip_gap=3.0,   # 45 deg lip over each module hides the LEDs from the camera (26 left the side walls visible)

    # Phase 2 electronics (docs/hardware/pcb_phase2.md): POWER board on the right (+X, beside the bank's
    # cable slot), LIGHT/IO board on the left, both flat on standoffs above the battery shelf
    pcb_main_x=(21.0, 67.0),      # |x| extent: clear of the bank cable slot (x 4..18) and the side wall
    pcb_main_z=(-32.0, 23.0),     # z extent: ends before the LED-pod screw bosses (z >= 23.8 above the board)
    pcb_low=(52.0, 61.0, 5.0),    # |x| strip under the display posts (x 56.5, 5.9 mm above the board): parts <= 5 mm
    pcb_t=1.6, pcb_standoff=5.0,  # board thickness; board underside above the shelf
    pcb_top=12.0, pcb_bot=2.0,    # component height above / below the board
    pcb_hole_in=3.5,              # M3 hole centres in from the board corners

    # cooling
    fan=(30.0, 7.0),              # 30 x 30 x 7 mm 5 V fan, exhausting up through the top vents

    # buttons and switches
    btn_d=12.2,                   # 12 mm panel push buttons (index-finger triggers)
    thumb_btn_d=7.2,              # 7 mm panel push buttons (thumbs; their nuts fit the arm root)
    pwr_btn_d=12.2, pwr_btn_x=45.0,   # 12 mm momentary power button with ring LED, top, user's left (POWER J5);
                                      # x clear of the display post at 56.5 for a 22 mm deep button body
    usb_a=(7.2, 14.5, 15.0), usb_a_x=-51.0,   # export port: panel-mount USB-A cut-out (x, z), M3 screws at +-15 mm (z);
                                              # top, user's right, on the flat (clear of the corner radius and the post)
    # charge-through port: USB-C through the right side wall by the bay (CHARGE board, docs/hardware/controls.md)
    chg_pcb=(24.0, 12.5, 1.6),    # board u along z, v along -x (mouth edge at the wall)
    chg_z0=-2.0, chg_y=-48.0,     # board u = 0 at z = chg_z0; board underside at y = chg_y
    chg_slot=(9.4, 3.8), chg_recess=(12.5, 7.0, 1.4),   # receptacle opening; outside recess for the plug overmould

    # fasteners (CNC Kitchen / ruthex heat-set inserts)
    m3_insert_d=4.0, m3_insert_l=6.7, m3_boss_d=7.6,
    m3_clear=3.4, m3_head_d=6.2, m3_cs_d=6.4, m3_pilot=2.5,
    m2_clear=2.4,
    m2_insert_d=3.2, m2_insert_l=4.0, m2_boss_d=5.5,   # ruthex M2 heat-set inserts
    btn_pcb_x=104.0, btn_pcb_y=12.0, btn_pcb_z=-1.0,   # SIDE board centre in the wing (|x|, model y, slab z')
)


# 5-inch display profiles. Positions are relative to the glass centre, in model X/Y as seen from
# the FRONT (+Z); z values are measured from the glass front face toward the inside.
#   disp_vis_off   visible-area centre offset        disp_tab    FPC tail below the glass (x0, x1, depth)
#   disp_back      glass front to the PCB/frame back disp_parts  back-side parts box (w, h, height)
#   disp_zif       tall DSI connector (x0, x1, y0, y1, z_max) or None
#   pi_under       glass front to the Pi PCB underside (the display's own standoffs)
#   pi_c_off       Pi hole-pattern centre;  pi_usb_dir: +1 USB/Ethernet toward +X (GPIO edge up), -1 rotated 180 deg
#   disp_posts     where the 4 retaining posts press;  disp_post_z: their contact face from the glass front
DISPLAYS = {
    # Waveshare 5inch DSI LCD (800x480): official drawing + 3D model 5inch-dsi-lcd_asm.stp (files.waveshare.com).
    # The Pi must sit with its USB/Ethernet end away from the tall DSI connector, hence pi_usb_dir = -1.
    'ws5_dsi': dict(
        disp_glass=(120.7, 77.2), disp_vis=(109.0, 65.8), disp_vis_off=(0.0, 2.6),
        disp_tab=(-42.6, -35.7, 1.6),
        disp_back=9.0, disp_pcb=(121.0, 76.0), disp_pcb_off=(0.0, 0.71), disp_parts=(115.0, 70.0, 2.0),
        disp_zif=(43.57, 48.66, -10.5, 11.9, 14.7),
        pi_under=13.0, pi_c_off=(-5.915, 0.71), pi_usb_dir=-1,
        disp_posts=((-56.5, 34.71), (56.5, 34.71), (-56.5, -33.29), (56.5, -33.29)), disp_post_z=13.0,
        body_w=147.0, x_split=86.0, grip_cx=112.0, pod_x_in=20.0),
    # Raspberry Pi Touch Display 2, 5" (SC1975), landscape: official product brief (glass front to the Pi
    # standoff tips is 15.9 mm). Posts press its 4 case-mount bracket blocks.
    'td2_5': dict(
        disp_glass=(143.4, 91.46), disp_vis=(111.4, 63.1), disp_vis_off=(0.0, 0.0),
        disp_tab=None,
        disp_back=7.4, disp_pcb=(122.74, 72.96), disp_pcb_off=(0.0, 0.0), disp_parts=(92.87, 32.6, 2.0),
        disp_zif=None,
        pi_under=15.9, pi_c_off=(-8.79, 0.0), pi_usb_dir=1,
        disp_posts=((-51.85, 25.5), (51.85, 25.5), (-51.85, -25.5), (51.85, -25.5)), disp_post_z=10.4,
        body_w=164.0, x_split=96.0, grip_cx=121.0, pod_x_in=22.0),
}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def V(mm):
    return adsk.core.ValueInput.createByReal(mm * MM)


def pt(x, y, z=0.0):
    return adsk.core.Point3D.create(x * MM, y * MM, z * MM)


def vec(x, y, z):
    return adsk.core.Vector3D.create(x, y, z)


def oc(items):
    c = adsk.core.ObjectCollection.create()
    for i in items:
        c.add(i)
    return c


class Geo:
    """Transient-solid toolkit (TemporaryBRepManager) working in mm."""

    def __init__(self):
        self.tbm = adsk.fusion.TemporaryBRepManager.get()

    def box(self, x0, x1, y0, y1, z0, z1):
        c = pt((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        obb = adsk.core.OrientedBoundingBox3D.create(
            c, vec(1, 0, 0), vec(0, 1, 0), abs(x1 - x0) * MM, abs(y1 - y0) * MM, abs(z1 - z0) * MM)
        return self.tbm.createBox(obb)

    def obox(self, frame, x0, x1, y0, y1, z0, z1):
        """Box spanning [x0,x1] x [y0,y1] x [z0,z1] in a local frame (origin, ex, ey, ez)."""
        o, ex, ey, ez = frame
        cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        c = [o[i] + cx * ex[i] + cy * ey[i] + cz * ez[i] for i in range(3)]
        obb = adsk.core.OrientedBoundingBox3D.create(pt(*c), vec(*ex), vec(*ey),
                                                     abs(x1 - x0) * MM, abs(y1 - y0) * MM, abs(z1 - z0) * MM)
        return self.tbm.createBox(obb)

    def cyl(self, a, b, r, r2=None):
        return self.tbm.createCylinderOrCone(pt(*a), r * MM, pt(*b), (r if r2 is None else r2) * MM)

    def ocyl(self, frame, a, b, r, r2=None):
        o, ex, ey, ez = frame
        g = lambda q: [o[i] + q[0] * ex[i] + q[1] * ey[i] + q[2] * ez[i] for i in range(3)]
        return self.cyl(g(a), g(b), r, r2)

    def union(self, bodies):
        bodies = [b for b in bodies if b is not None]
        if not bodies:
            return None
        res = self.tbm.copy(bodies[0])
        for b in bodies[1:]:
            self.tbm.booleanOperation(res, b, adsk.fusion.BooleanTypes.UnionBooleanType)
        return res

    def intersect(self, a, b):
        res = self.tbm.copy(a)
        self.tbm.booleanOperation(res, b, adsk.fusion.BooleanTypes.IntersectionBooleanType)
        return res

    def subtract(self, a, b):
        res = self.tbm.copy(a)
        self.tbm.booleanOperation(res, b, adsk.fusion.BooleanTypes.DifferenceBooleanType)
        return res


def add_temp_body(comp, tbody, name):
    bf = comp.features.baseFeatures.add()
    bf.startEdit()
    comp.bRepBodies.add(tbody, bf)
    bf.finishEdit()
    body = bf.bodies.item(bf.bodies.count - 1)
    body.name = name
    return body


def combine(comp, target, tool, op):
    ci = comp.features.combineFeatures.createInput(target, oc([tool]))
    ci.operation = op
    ci.isKeepToolBodies = False
    comp.features.combineFeatures.add(ci)


def apply_details(comp, target, adds, cuts, tag):
    g = Geo()
    if adds:
        combine(comp, target, add_temp_body(comp, g.union(adds), tag + '_adds'),
                adsk.fusion.FeatureOperations.JoinFeatureOperation)
    if cuts:
        combine(comp, target, add_temp_body(comp, g.union(cuts), tag + '_cuts'),
                adsk.fusion.FeatureOperations.CutFeatureOperation)


def sketch_poly(comp, pts, radii):
    """Closed polyline in the XY plane, with a fillet radius per vertex (0 = sharp)."""
    sk = comp.sketches.add(comp.xYConstructionPlane)
    L = sk.sketchCurves.sketchLines
    n = len(pts)
    lines = []
    for i in range(n):
        b = pts[(i + 1) % n]
        if i == 0:
            ln = L.addByTwoPoints(pt(*pts[0]), pt(*b))
        elif i == n - 1:
            ln = L.addByTwoPoints(lines[-1].endSketchPoint, lines[0].startSketchPoint)
        else:
            ln = L.addByTwoPoints(lines[-1].endSketchPoint, pt(*b))
        lines.append(ln)
    A = sk.sketchCurves.sketchArcs
    for i in range(n):
        if radii[i] > 0:
            a_ln, b_ln = lines[i - 1], lines[i]
            A.addFillet(a_ln, a_ln.endSketchPoint.geometry, b_ln, b_ln.startSketchPoint.geometry, radii[i] * MM)
    return sk


def extrude_new(comp, sk, z0, z1, name):
    prof = max((sk.profiles.item(i) for i in range(sk.profiles.count)),
               key=lambda p: p.areaProperties().area)
    ext = comp.features.extrudeFeatures
    inp = ext.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    inp.startExtent = adsk.fusion.OffsetStartDefinition.create(V(z0))
    inp.setDistanceExtent(False, V(z1 - z0))
    body = ext.add(inp).bodies.item(0)
    body.name = name
    return body


def edges_at_z(body, z, tol=1e-4):
    out = []
    for e in body.edges:
        bb = e.boundingBox
        if abs(bb.minPoint.z - z * MM) < tol and abs(bb.maxPoint.z - z * MM) < tol:
            out.append(e)
    return out


def fillet(comp, edges, r):
    fi = comp.features.filletFeatures.createInput()
    fi.edgeSetInputs.addConstantRadiusEdgeSet(oc(edges), V(r), True)
    comp.features.filletFeatures.add(fi)


def offset_plane(comp, base, d, axis):
    """Plane parallel to `base` at coordinate d along `axis` (0=x, 1=y, 2=z), whatever the base normal sign."""
    pi = comp.constructionPlanes.createInput()
    pi.setByOffset(base, V(d))
    pl = comp.constructionPlanes.add(pi)
    o = pl.geometry.origin
    got = (o.x, o.y, o.z)[axis] / MM
    if abs(got - d) > 1e-3:                      # base normal points the other way
        pl.deleteMe()
        pi = comp.constructionPlanes.createInput()
        pi.setByOffset(base, V(-d))
        pl = comp.constructionPlanes.add(pi)
    pl.isLightBulbOn = False
    return pl


def split(comp, body, plane):
    sp = comp.features.splitBodyFeatures
    f = sp.add(sp.createInput(body, plane, True))
    return [f.bodies.item(i) for i in range(f.bodies.count)]


def bbox_mm(body):
    bb = body.boundingBox
    return [round(v / MM, 1) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z,
                                       bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]


# ----------------------------------------------------------------------------------------------
# derived layout
# ----------------------------------------------------------------------------------------------
def layout(p):
    d = dict(p)
    t = p['wall']
    d['bx'] = p['body_w'] / 2
    d['sz'] = p['slab_t'] / 2
    d['szc'] = p['slab_zc']
    d['rake'] = math.radians(p['grip_rake_deg'])
    d['t_rake'] = math.tan(d['rake'])
    d['sf'] = d['sz'] * math.cos(d['rake'])            # slab faces at local z' = +-sf (slab frame)
    d['sw'] = -4.0 * math.cos(d['rake'])               # WingBack | WingFront split plane, local z'
    d['grip'] = grip_outline(d)
    d['zf_in'] = p['body_z_front'] - t
    d['y_top_in'] = p['body_y_top'] - t
    cy = p['disp_cy']
    # display + Pi stack, from the back
    d['z_glass'] = p['body_z_back'] + p['disp_lip']
    d['z_disp_back'] = d['z_glass'] + p['disp_back']
    d['z_pcb0'] = d['z_glass'] + p['pi_under']
    d['z_pcb1'] = d['z_pcb0'] + p['pi_pcb_t']
    d['z_hat0'] = d['z_pcb1'] + p['hat_standoff']
    d['z_hat1'] = d['z_hat0'] + p['hat_pcb_t']
    d['z_hs1'] = d['z_pcb1'] + p['stack_top']
    d['pi_cx'] = p['pi_c_off'][0]
    d['pi_cy'] = cy + p['pi_c_off'][1]
    # camera: lens on (0, cy); FPC edge points down
    d['z_lens'] = p['body_z_front'] - p['cam_set_back']
    d['z_cam_front'] = d['z_lens'] - p['cam_lens_h']
    d['z_cam_back'] = d['z_cam_front'] - p['cam_pcb_t'] - 2.75 - 0.5
    d['cam_top'] = cy + p['cam_lens_from_far']
    # battery bay: bank lying flat along X on the BayDoor floor, plug zone toward +X
    d['bay_y0'] = p['body_y_bot'] + t
    d['bay_y1'] = d['bay_y0'] + p['bay_t']
    d['shelf_y1'] = d['bay_y1'] + 2.0
    d['bay_x0'] = -(p['bay_l'] + p['bay_plug']) / 2
    d['bay_x1'] = (p['bay_l'] + p['bay_plug']) / 2
    d['bay_zc'] = (p['body_z_back'] + p['body_z_front']) / 2
    # LED pod footprint (X > 0 side)
    d['pod_x0'] = p['pod_x_in']
    d['pod_x1'] = d['bx'] - p['body_r_edge'] - p['pod_margin']
    d['pod_y0'] = p['body_y_bot'] + p['body_r_edge'] + p['pod_margin']
    d['pod_y1'] = p['body_y_top'] - p['body_r_edge'] - p['pod_margin']
    d['pod_face_w'] = (d['pod_x1'] - d['pod_x0']) / math.cos(math.radians(p['toe_deg']))
    # pod rows, top to bottom
    d['ytop'] = d['pod_y1'] - 5.4                      # top of the white strips (wire slots above)
    d['y_wtop'] = d['ytop'] + 3.0                      # WHITE carrier top edge (v = 0)
    d['y_wb'] = d['y_wtop'] - p['white_pcb'][1]        # carrier bottom edge
    d['y_screw'] = d['ytop'] - 78.0 - 7.5              # pod screws (fixed: the UV block and the body windows stay put)
    d['y_uv'] = d['y_screw'] - 3.1 - 0.8 - 1.6         # top of the UV board area (frame inside)
    d['uv_h'] = p['uv_board'][0] + 0.8
    d['uv_cav_w'] = (p['uv_n'] - 1) * p['uv_col_gap'] + p['uv_board'][1] + 1.8
    # fan under the top vents
    d['fan_cx'] = d['pi_cx']
    d['fan_cz'] = d['bay_zc']
    return d



def slab_frame(d):
    """Frame of the raked wing slab: x across, y' along the raked long axis, z' out of the slab front."""
    c, sn = math.cos(d['rake']), math.sin(d['rake'])
    return ((0.0, 0.0, d['szc']), (1.0, 0.0, 0.0), (0.0, c, sn), (0.0, -sn, c))


def sy(d, y):
    """Model y of a point on the slab mid-plane -> slab-local y'."""
    return y / math.cos(d['rake'])


def z_mid(d, y):
    """Model z of the slab mid-plane at height y."""
    return d['szc'] + y * d['t_rake']


def grip_xy(d, s, lx, ly):
    """Grip-local (lx outward, ly up) -> model (x, y), with the splay (tops lean inward)."""
    a = s * math.radians(d['grip_splay_deg'])
    x, y = s * lx, ly
    return (s * d['grip_cx'] + x * math.cos(a) - y * math.sin(a), d['grip_cy'] + x * math.sin(a) + y * math.cos(a))


def grip_outline(d):
    """Right-hand (X > 0) wing outline from the arm root around the grip, with a radius per vertex."""
    s = 1
    L, we, wm = d['grip_len'], d['grip_w_end'], d['grip_w_mid']
    IT, OT, OM = grip_xy(d, s, -we / 2, L / 2), grip_xy(d, s, we / 2, L / 2), grip_xy(d, s, wm / 2, 0)
    OB, IB, IM = grip_xy(d, s, we / 2, -L / 2), grip_xy(d, s, -we / 2, -L / 2), grip_xy(d, s, -wm / 2, 0)

    def cross_y(a, b, y0):
        t = (y0 - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y0), t

    p_top, t1 = cross_y(IT, IM, d['arm_y_top'])
    p_bot, t2 = cross_y(IM, IB, d['arm_y_bot'])
    xi = d['bx'] - 10                                   # arms start inside the body
    pts = [(xi, d['arm_y_top']), p_top, IT, OT, OM, OB, IB, p_bot, (xi, d['arm_y_bot'])]
    rad = [0, d['arm_r_corner'], d['grip_r_inner'], d['grip_r_end'], 400.0, d['grip_r_end'],
           d['grip_r_inner'], d['arm_r_corner'], 0]
    return dict(pts=pts, rad=rad, ok=(0 < t1 < 1 and 0 < t2 < 1),
                inner_min_x=min(IT[0], IB[0], p_top[0], p_bot[0]),
                outer_max_x=max(OT[0], OB[0], OM[0]))


def edges_on_slab_face(body, d, zloc, tol=0.02):
    """Edges lying on the raked slab face local z' = zloc (mm)."""
    o, ex, ey, ez = slab_frame(d)
    out = []
    for e in body.edges:
        ev = e.evaluator
        ok, p0, p1 = ev.getParameterExtents()
        on = True
        for tp in (p0, (p0 + p1) / 2, p1):
            ok2, q = ev.getPointAtParameter(tp)
            zl = (q.x / MM - o[0]) * ez[0] + (q.y / MM - o[1]) * ez[1] + (q.z / MM - o[2]) * ez[2]
            if abs(zl - zloc) > tol:
                on = False
                break
        if on:
            out.append(e)
    return out


def tilted_plane(comp, d, z_off):
    """Construction plane z = szc + z_off + y*tan(rake), i.e. the slab mid-plane shifted by z_off along Z."""
    def make(angle):
        pi = comp.constructionPlanes.createInput()
        pi.setByAngle(comp.xConstructionAxis, adsk.core.ValueInput.createByReal(angle), comp.xYConstructionPlane)
        return comp.constructionPlanes.add(pi)
    base = make(d['rake'])
    n = base.geometry.normal
    if abs(n.y / n.z + d['t_rake']) > 1e-6:            # rotated the wrong way: use the other sense
        base.deleteMe()
        base = make(-d['rake'])
        n = base.geometry.normal
    z0 = d['szc'] + z_off
    off = z0 * math.cos(d['rake']) * (1 if n.z > 0 else -1)
    for sign in (1, -1):
        pi = comp.constructionPlanes.createInput()
        pi.setByOffset(base, V(sign * off))
        pl = comp.constructionPlanes.add(pi)
        g = pl.geometry
        o, nn = g.origin, g.normal
        if abs((0 - o.x) * nn.x + (0 - o.y) * nn.y + (z0 * MM - o.z) * nn.z) < 1e-6:
            break
        pl.deleteMe()
    base.isLightBulbOn = False
    pl.isLightBulbOn = False
    return pl


def checks(d):
    """Refuse to build an impossible layout; return a list of messages."""
    msgs = []
    t = d['wall']
    inner_w = d['body_w'] - 2 * t
    inner_d = (d['body_z_front'] - d['body_z_back']) - 2 * t
    gw, gh = d['disp_glass']
    if d['bay_x1'] - d['bay_x0'] + 4 > inner_w:
        msgs.append('battery bay too long for the body')
    if d['bay_w'] + 1.0 > inner_d:
        msgs.append('battery bay too wide for the depth: %.1f > %.1f' % (d['bay_w'] + 1, inner_d))
    if gw / 2 + 0.25 > d['bx'] - t - 3.2 - d['m3_boss_d'] / 2:
        msgs.append('display glass leaves no room for the lid screw bosses at the sides')
    if d['disp_cy'] + gh / 2 + 0.3 > d['y_top_in']:
        msgs.append('display glass hits the top wall')
    if d['pi_cy'] - 28 < d['shelf_y1'] + 8:
        msgs.append('Pi too close to the battery shelf / electronics carriers')
    if d['z_hs1'] + 2 > d['z_cam_back']:
        msgs.append('HAT heatsink pins (z %.1f) too close to the camera back (z %.1f)' % (d['z_hs1'], d['z_cam_back']))
    white = d['white_n'] * d['white_w'] + (d['white_n'] - 1) * d['white_gap']
    if white + 4 > d['pod_face_w']:
        msgs.append('white strips (%.1f) too wide for the pod face (%.1f)' % (white, d['pod_face_w']))
    uv_bottom = d['y_uv'] - d['uv_h'] * math.cos(math.radians(d['uv_tilt_deg'])) - 1.6
    if uv_bottom < d['pod_y0'] + 0.5:
        msgs.append('UV block runs off the bottom of the pod by %.1f mm' % (d['pod_y0'] + 0.5 - uv_bottom))
    if d['x_split'] < d['bx'] + 12:
        msgs.append('x_split must leave room for the thumb buttons on the centre piece')
    if not d['grip']['ok']:
        msgs.append('arm top/bottom must meet the grip inner edge: adjust arm_y_top/arm_y_bot or grip_len')
    if d['grip']['inner_min_x'] < d['x_split'] + 3:
        msgs.append('grip inner edge (%.1f) too close to the wing split (%.1f): reduce splay or raise grip_cx'
                    % (d['grip']['inner_min_x'], d['x_split']))
    if d['door_y'] < d['bay_y0'] + 1 or d['door_y'] > d['bay_y1'] - 10:
        msgs.append('door_y must cut through the bay walls')
    return msgs


# ----------------------------------------------------------------------------------------------
# the build
# ----------------------------------------------------------------------------------------------
PRINTED = ('FrontCentre', 'BackLid', 'BayDoor', 'CamHood', 'LedPod_L', 'LedPod_R', 'InspectionHood',
           'WingFront_L', 'WingFront_R', 'WingBack_L', 'WingBack_R')


def build(overrides=None, export=True, out_dir=OUT_DIR):
    overrides = overrides or {}
    p = dict(P)
    p.update(DISPLAYS[overrides.get('display', P['display'])])
    p.update(overrides)
    d = layout(p)
    problems = checks(d)
    if problems:
        raise ValueError('Layout problems:\n  ' + '\n  '.join(problems))

    app = adsk.core.Application.get()
    app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    des = adsk.fusion.Design.cast(app.activeProduct)
    des.designType = adsk.fusion.DesignTypes.ParametricDesignType
    root = des.rootComponent
    enc = root.occurrences.addNewComponent(adsk.core.Matrix3D.create()).component
    enc.name = 'Enclosure'
    ref = root.occurrences.addNewComponent(adsk.core.Matrix3D.create()).component
    ref.name = 'Reference (not printed)'

    parts = split_shell(enc, build_outer_shell(enc, d), d)
    detail_front_centre(enc, parts['FrontCentre'], d)
    detail_back_lid(enc, parts['BackLid'], d)
    detail_bay_door(enc, parts['BayDoor'], d)
    for side in ('L', 'R'):
        s = 1 if side == 'L' else -1          # the user's left hand is +X
        detail_wing_front(enc, parts['WingFront_' + side], d, s)
        detail_wing_back(enc, parts['WingBack_' + side], d, s)
        build_led_pod(enc, d, s, 'LedPod_' + side)
    build_cam_hood(enc, d)
    if d['hood_on']:
        build_inspection_hood(enc, d)
        hood_mount_front_centre(enc, parts['FrontCentre'], d)
    build_reference(ref, d)
    if d['hood_on']:
        build_hood_reference(ref, d)

    stray = [b for b in enc.bRepBodies if b.name not in PRINTED]
    print(interference_report(des, enc, ref))
    total = 0.0
    for b in enc.bRepBodies:
        total += b.volume
        print('part %-13s bbox(mm) %s  %.0f cm3' % (b.name, bbox_mm(b), b.volume))
    print('stray bodies:', [b.name for b in stray] or 'none')
    print('plastic %.0f cm3 -> about %.0f g in PETG with solid walls' % (total, total * 1.27))
    print('overall width %.0f mm; grips %.0f mm centre to centre, rake %.0f deg, splay %.0f deg; '
          'wing joining screws M3 x %.0f'
          % (2 * d['grip']['outer_max_x'], 2 * d['grip_cx'], d['grip_rake_deg'], d['grip_splay_deg'],
             d['sw'] + d['m3_insert_l'] - 1.0 + d['sf'] - 3.2))
    set_view(app)
    if export:
        export_all(des, enc, out_dir)
    return d


# ---------------------------------------------------------------- outer shell
def build_outer_shell(comp, d):
    bx = d['bx']
    yb, yt = d['body_y_bot'], d['body_y_top']
    rt, rb = d['body_r_top'], d['body_r_bot']
    sk = sketch_poly(comp, [(-bx, yb), (bx, yb), (bx, yt), (-bx, yt)], [rb, rb, rt, rt])
    block = extrude_new(comp, sk, d['body_z_back'], d['body_z_front'], 'block')
    fillet(comp, edges_at_z(block, d['body_z_front']) + edges_at_z(block, d['body_z_back']), d['body_r_edge'])

    # arms + grips: right outline from the arm root around the grip, then the mirrored left side.
    # Extrude tall, then intersect with the raked slab so the whole wing leans by the rake angle.
    right = d['grip']['pts']
    rad_r = d['grip']['rad']
    left = [(-x, y) for (x, y) in reversed(right)]
    sk2 = sketch_poly(comp, right + left, rad_r + list(reversed(rad_r)))
    slab = extrude_new(comp, sk2, d['szc'] - 60, d['szc'] + 60, 'slab')
    g = Geo()
    raked = add_temp_body(comp, g.obox(slab_frame(d), -400, 400, -300, 300, -d['sf'], d['sf']), 'raked')
    combine(comp, slab, raked, adsk.fusion.FeatureOperations.IntersectFeatureOperation)
    fillet(comp, edges_on_slab_face(slab, d, d['sf']) + edges_on_slab_face(slab, d, -d['sf']), d['slab_r_edge'])

    combine(comp, block, slab, adsk.fusion.FeatureOperations.JoinFeatureOperation)
    sh = comp.features.shellFeatures
    si = sh.createInput(oc([block]), False)
    si.insideThickness = V(d['wall'])
    sh.add(si)
    block.name = 'shell'
    return block


def split_shell(comp, body, d):
    xs = d['x_split']
    bodies = split(comp, body, offset_plane(comp, comp.yZConstructionPlane, xs, 0))
    plus = [b for b in bodies if b.boundingBox.minPoint.x > (xs - 1) * MM][0]
    rest = [b for b in bodies if b.boundingBox.minPoint.x < (xs - 1) * MM][0]
    bodies = split(comp, rest, offset_plane(comp, comp.yZConstructionPlane, -xs, 0))
    minus = [b for b in bodies if b.boundingBox.maxPoint.x < (-xs + 1) * MM][0]
    centre = [b for b in bodies if b.boundingBox.maxPoint.x > (-xs + 1) * MM][0]
    parts = {}
    bodies = split(comp, centre, offset_plane(comp, comp.xZConstructionPlane, d['door_y'], 1))
    door = [b for b in bodies if b.boundingBox.maxPoint.y < (d['door_y'] + 0.5) * MM][0]
    upper = [b for b in bodies if b is not door and b.boundingBox.maxPoint.y > (d['door_y'] + 0.5) * MM][0]
    parts['BayDoor'] = door
    for b in split(comp, upper, offset_plane(comp, comp.xYConstructionPlane, d['z_split_centre'], 2)):
        parts['BackLid' if b.boundingBox.maxPoint.z < (d['z_split_centre'] + 0.5) * MM else 'FrontCentre'] = b
    pw = tilted_plane(comp, d, d['sw'] / math.cos(d['rake']))
    o, ex, ey, ez = slab_frame(d)
    for side, wb in (('L', plus), ('R', minus)):          # +X is the user's LEFT hand
        for b in split(comp, wb, pw):
            c = b.physicalProperties.centerOfMass
            zl = (c.x / MM - o[0]) * ez[0] + (c.y / MM - o[1]) * ez[1] + (c.z / MM - o[2]) * ez[2]
            parts[('WingBack_' if zl < d['sw'] else 'WingFront_') + side] = b
    for k, b in parts.items():
        b.name = k
    return parts


# ---------------------------------------------------------------- shared positions
def lid_screw_points(d):
    x = d['bx'] - d['wall'] - 3.2
    cy = d['disp_cy']
    return [(sx * x, cy + dy) for sx in (-1, 1) for dy in (-22, 22)]


def pi_xy(d, x, y):
    """Pi-local (x, y) -> model (X, Y). The 58 x 49 hole pattern is centred at local (32.5, 28)."""
    u = d['pi_usb_dir']
    return d['pi_cx'] + u * (x - 32.5), d['pi_cy'] + u * (y - 28.0)


def pi_box(g, d, x0, x1, y0, y1, z0, z1):
    ax, ay = pi_xy(d, x0, y0)
    bx_, by = pi_xy(d, x1, y1)
    return g.box(min(ax, bx_), max(ax, bx_), min(ay, by), max(ay, by), z0, z1)


def display_post_points(d):
    return [(x, d['disp_cy'] + y) for (x, y) in d['disp_posts']]


def cam_points(d):
    """Camera mounting holes; board centred on X = 0, far edge up, FPC edge down."""
    return [(-d['cam_w'] / 2 + hx, d['cam_top'] - hy) for (hx, hy) in d['cam_holes_far']]


def pod_screw_points(d, s):
    xc = (d['pod_x0'] + d['pod_x1']) / 2
    return [(s * (xc - 13), d['y_screw']), (s * (xc + 13), d['y_screw'])]


def pcb_main_box(d, s):
    """(x0, x1, z0, z1) of the POWER (s = +1) or LIGHT/IO (s = -1) board outline."""
    a, b = d['pcb_main_x']
    x0, x1 = (a, b) if s > 0 else (-b, -a)
    return x0, x1, d['pcb_main_z'][0], d['pcb_main_z'][1]


def elec_points(d):
    """(x, z) of the bosses standing on the battery shelf: the 4 corner holes of each main board
    (docs/hardware/pcb_phase2.md)."""
    k, out = d['pcb_hole_in'], []
    for s in (1, -1):
        x0, x1, z0, z1 = pcb_main_box(d, s)
        out += [(x, z) for x in (x0 + k, x1 - k) for z in (z0 + k, z1 - k)]
    return out


def door_screw_points(d):
    zc = d['bay_zc']
    return [(x, zc + dz) for x in (d['bay_x0'] - 5.4, d['bay_x1'] + 5.4) for dz in (-20.0, 20.0)]


def wing_screw_points(d):
    """Model (x, y) of the WingFront/WingBack joining screws (right side), near the grip ends."""
    L = d['grip_len']
    return [grip_xy(d, 1, 0.0, L / 2 - 14.0), grip_xy(d, 1, 0.0, -L / 2 + 14.0)]


def thumb_button_points(d):
    """On the flat part of the arm-root back face (the edge rounding starts near y = +-29)."""
    xa = (d['bx'] + d['x_split']) / 2
    return [(xa, 24.0), (xa, 8.0)]


def back_tab_screw_points(d):
    """Back joint tab screws (right side): through the WingBack skin into the centre piece."""
    return [(d['x_split'] + 4.0, -21.0), (d['x_split'] + 4.0, -11.0)]


def tab_screw_x(d):
    return d['x_split'] + 6


# ---------------------------------------------------------------- front centre
def detail_front_centre(comp, body, d):
    g = Geo()
    t, bx = d['wall'], d['bx']
    zf, zf_in, zs = d['body_z_front'], d['zf_in'], d['z_split_centre']
    cy = d['disp_cy']
    adds, cuts = [], []

    # camera bosses (M2 self-tapping) and the lens/tool hole
    for (x, y) in cam_points(d):
        adds.append(g.cyl((x, y, d['z_cam_front']), (x, y, zf_in + 0.5), d['cam_boss_d'] / 2))
        cuts.append(g.cyl((x, y, d['z_cam_front'] - 0.1), (x, y, d['z_cam_front'] + d['m2_insert_l']), d['m2_insert_d'] / 2))
    cuts.append(g.cyl((0, cy, zf_in - 1), (0, cy, zf + 1), d['tool_hole_d'] / 2))
    # hood screw pilots on lugs above and below the hood
    for dy in (-d['hood_screw_dy'], d['hood_screw_dy']):
        adds.append(g.cyl((0, cy + dy, zf_in - 4), (0, cy + dy, zf_in + 0.5), 2.9))
        cuts.append(g.cyl((0, cy + dy, zf - d['m2_insert_l']), (0, cy + dy, zf + 1), d['m2_insert_d'] / 2))

    # display posts: from the front wall to the display's case-mount bracket blocks (1 mm foam pad).
    # They hold the display + Pi + HAT assembly against the BackLid lip; nothing presses on the HAT.
    for (x, y) in display_post_points(d):
        z_tip = d['z_glass'] + d['disp_post_z'] + d['post_pad']
        adds.append(g.cyl((x, y, z_tip), (x, y, zf_in + 0.5), d['post_d'] / 2))

    # battery shelf, bay end stops with BayDoor screw bosses, cable slot at the plug end
    adds.append(g.box(-(bx - t) - 0.5, bx - t + 0.5, d['bay_y1'], d['shelf_y1'], zs, zf_in + 0.5))
    for xe in (d['bay_x0'] - 2, d['bay_x1']):
        adds.append(g.box(xe, xe + 2, d['door_y'], d['bay_y1'] + 0.5, zs, zf_in + 0.5))
    for (x, z) in door_screw_points(d):
        adds.append(g.cyl((x, d['door_y'], z), (x, d['door_y'] + 9, z), d['m3_boss_d'] / 2))
        cuts.append(g.cyl((x, d['door_y'] - 0.1, z), (x, d['door_y'] + d['m3_insert_l'], z), d['m3_insert_d'] / 2))
    cuts.append(g.box(4, 18, d['bay_y1'] - 1, d['shelf_y1'] + 1, d['bay_zc'] - 8, d['bay_zc'] + 8))     # bank cable up

    # BackLid screw bosses (inserts face the split plane)
    for (x, y) in lid_screw_points(d):
        adds.append(g.cyl((x, y, zs), (x, y, zf_in + 0.5), d['m3_boss_d'] / 2))           # full-depth column
        cuts.append(g.cyl((x, y, zs - 0.1), (x, y, zs + d['m3_insert_l']), d['m3_insert_d'] / 2))

    # LED pod mounting bosses and the wire windows behind each pod
    for s in (1, -1):
        for (x, y) in pod_screw_points(d, s):
            adds.append(g.cyl((x, y, zf_in - 8), (x, y, zf_in + 0.5), d['m3_boss_d'] / 2))
            cuts.append(g.cyl((x, y, zf - d['m3_insert_l']), (x, y, zf + 0.1), d['m3_insert_d'] / 2))
        xa, xb = s * (d['pod_x0'] + 4), s * (d['pod_x1'] - 4)
        x0, x1 = min(xa, xb), max(xa, xb)
        cuts.append(g.box(x0, x1, d['ytop'] - 1, d['ytop'] + 3.8, zf_in - 1, zf + 1))      # white, top ends
        cuts.append(g.box(x0, x1, d['pod_y0'] + 2, d['y_uv'] + 0.5, zf_in - 1, zf + 1))   # UV boards

    # power-electronics carrier bosses on the battery shelf
    ys = d['shelf_y1']
    for (x, z) in elec_points(d):                      # M3 x 6 into heat-set inserts (boards come out any number of times)
        adds.append(g.cyl((x, ys - 0.5, z), (x, ys + d['pcb_standoff'], z), d['m3_boss_d'] / 2))   # board sits on the boss tops
        cuts.append(g.cyl((x, ys + d['pcb_standoff'] - 5.8, z), (x, ys + d['pcb_standoff'] + 0.1, z), d['m3_insert_d'] / 2))   # short M3 x 5.7 insert

    # 30 mm exhaust fan: 4 bosses hanging from the top wall (M3 self-tapping, 24 mm pattern)
    fs, ft = d['fan']
    for dx in (-12, 12):
        for dz in (-12, 12):
            x, z = d['fan_cx'] + dx, d['fan_cz'] + dz
            adds.append(g.cyl((x, d['y_top_in'] - 6, z), (x, d['y_top_in'] + 0.5, z), 2.8))
            cuts.append(g.cyl((x, d['y_top_in'] - 6.1, z), (x, d['y_top_in'] - 0.5, z), 1.25))

    # wing joints: tabs on the centre piece reaching into each arm. The front tab follows the raked
    # front skin (slab frame); the top/bottom tabs sit on the arm's horizontal walls, centred on the slab.
    F, sf = slab_frame(d), d['sf']
    xsc = tab_screw_x(d)
    for s in (1, -1):
        xa, xb = s * (bx - 1), s * (d['x_split'] + 10)
        x0, x1 = min(xa, xb), max(xa, xb)
        adds.append(g.obox(F, x0, x1, -14, 14, sf - t - 6, sf - t))
        # top/bottom tabs: fused to the wall inside the centre piece, 0.3 mm clear of the wing walls beyond
        xr, xe = s * (bx - 1), s * d['x_split']
        xm0, xm1 = min(xr, xe), max(xr, xe)
        ya, yb = d['arm_y_top'] - t, d['arm_y_bot'] + t
        za, zb = z_mid(d, ya), z_mid(d, yb)
        # (locating keys only: beyond the split the arm top/bottom blend into the grip)
        xk = s * (d['x_split'] + 5)
        xk0, xk1 = min(xr, xk), max(xr, xk)
        adds.append(g.box(xm0, xm1, ya - 6, ya + 0.5, za - 6, za + 6))
        adds.append(g.box(xk0, xk1, ya - 6, ya - 0.3, za - 6, za + 6))
        adds.append(g.box(xm0, xm1, yb - 0.5, yb + 6, zb - 6, zb + 6))
        adds.append(g.box(xk0, xk1, yb + 0.3, yb + 6, zb - 6, zb + 6))
        for yy in (-7, 7):
            cuts.append(g.ocyl(F, (s * xsc, yy, sf - t - d['m3_insert_l']), (s * xsc, yy, sf - t + 0.1), d['m3_insert_d'] / 2))
        # back tab under the arm's back skin, screwed from the WingBack side
        adds.append(g.obox(F, x0, x1, -25, -7, -sf + t, -sf + t + 6))
        for (x, y) in back_tab_screw_points(d):
            yl = sy(d, y)
            cuts.append(g.ocyl(F, (s * x, yl, -sf + t - 0.1), (s * x, yl, -sf + t + d['m3_insert_l']), d['m3_insert_d'] / 2))
        # thumb buttons on the back of the arm root, beside the display, square to the raked face
        for (x, y) in thumb_button_points(d):
            yl = sy(d, y)
            cuts.append(g.ocyl(F, (s * x, yl, -sf - 1), (s * x, yl, -sf + t + 1), d['thumb_btn_d'] / 2))

    # vents. Intake: front face, camera column above and below the hood.
    # Exhaust: top face, over the fan.
    for i in range(-3, 4):
        x = 4.0 * i
        cuts.append(g.box(x - 1.0, x + 1.0, d['shelf_y1'] + 4, cy - 25, zf_in - 1, zf + 1))
    for i in range(-4, 5):
        x = 4.0 * i
        cuts.append(g.box(x - 1.0, x + 1.0, cy + 25, d['body_y_top'] - d['body_r_edge'] - 2, zf_in - 1, zf + 1))
    for i in range(12):
        x = d['pi_cx'] - 33 + 6 * i
        cuts.append(g.box(x - 1.0, x + 1.0, d['y_top_in'] - 1, d['body_y_top'] + 1, d['fan_cz'] - 22, d['fan_cz'] + 22))

    # top: power button (user's left = +X, above the POWER board) and the USB-A export port (user's right = -X)
    px_, ux0 = d['pwr_btn_x'], d['usb_a_x']
    cuts.append(g.cyl((px_, d['y_top_in'] - 1, d['bay_zc']), (px_, d['body_y_top'] + 1, d['bay_zc']), d['pwr_btn_d'] / 2))
    ux, uz, us = d['usb_a']
    cuts.append(g.box(ux0 - ux / 2, ux0 + ux / 2, d['y_top_in'] - 1, d['body_y_top'] + 1, d['bay_zc'] - uz / 2, d['bay_zc'] + uz / 2))
    for dz in (-us, us):
        cuts.append(g.cyl((ux0, d['y_top_in'] - 1, d['bay_zc'] + dz), (ux0, d['body_y_top'] + 1, d['bay_zc'] + dz), d['m3_clear'] / 2))
    # charge-through port: USB-C opening + outside recess in the right side wall, a rib that carries the CHARGE
    # board (M2 into the rib from below), and a cable hole up through the shelf to POWER J6
    cw, ch_, ct = d['chg_pcb']
    zc = d['chg_z0'] + 6.0                                   # receptacle centre (board u = 6)
    yc = d['chg_y'] + ct + 1.63                              # receptacle centre height above the board top
    sw_, sh_ = d['chg_slot']
    rw_, rh_, rd_ = d['chg_recess']
    cuts.append(g.box(bx - t - 1, bx + 1, yc - sh_ / 2, yc + sh_ / 2, zc - sw_ / 2, zc + sw_ / 2))
    cuts.append(g.box(bx - rd_, bx + 1, yc - rh_ / 2, yc + rh_ / 2, zc - rw_ / 2, zc + rw_ / 2))
    hx, hz = bx - t - 0.2 - 6.25, d['chg_z0'] + 21.5          # board hole (u 21.5, v 6.25)
    adds.append(g.box(hx - 2.8, bx - t + 0.5, d['chg_y'] + ct, d['shelf_y1'] - 2.0, hz - 2.8, hz + 2.8))
    cuts.append(g.cyl((hx, d['chg_y'] + ct - 0.1, hz), (hx, d['chg_y'] + ct + d['m2_insert_l'], hz), d['m2_insert_d'] / 2))
    cuts.append(g.box(bx - t - 12.0, bx - t - 6.0, d['bay_y1'] - 1, d['shelf_y1'] + 1, d['chg_z0'] + 13.0, d['chg_z0'] + 20.0))

    apply_details(comp, body, adds, cuts, 'FrontCentre')


# ---------------------------------------------------------------- back lid (display)
def detail_back_lid(comp, body, d):
    g = Geo()
    zb, zs, t = d['body_z_back'], d['z_split_centre'], d['wall']
    cy = d['disp_cy']
    adds, cuts = [], []
    vw, vh = d['disp_vis']
    vx, vy = d['disp_vis_off']
    cuts.append(g.box(vx - vw / 2 - 0.5, vx + vw / 2 + 0.5, cy + vy - vh / 2 - 0.5, cy + vy + vh / 2 + 0.5, zb - 1, zb + t + 1))
    gw, gh = d['disp_glass']
    cuts.append(g.box(-gw / 2 - 0.25, gw / 2 + 0.25, cy - gh / 2 - 0.25, cy + gh / 2 + 0.25, d['z_glass'], zs + 1))
    if d['disp_tab']:
        tx0, tx1, tdep = d['disp_tab']
        cuts.append(g.box(tx0 - 0.5, tx1 + 0.5, cy - gh / 2 - tdep, cy - gh / 2, d['z_glass'], zs + 1))
    for (x, y) in lid_screw_points(d):
        adds.append(g.cyl((x, y, zb + t - 0.5), (x, y, zs), d['m3_boss_d'] / 2))
        cuts.append(g.cyl((x, y, zb - 1), (x, y, zs + 0.1), d['m3_clear'] / 2))
        cuts.append(g.cyl((x, y, zb - 1), (x, y, zb + 3.2), d['m3_head_d'] / 2))
    # locating lip: 1.2 x 2 mm strips that slip 0.3 mm inside the FrontCentre walls
    bx = d['bx']
    yti = d['y_top_in']
    # each strip has a root fused to the lid wall (below the split) and a lip above it
    adds.append(g.box(-(bx - 16), bx - 16, yti - 1.5, yti + 0.5, zs - 3.0, zs))
    adds.append(g.box(-(bx - 16), bx - 16, yti - 1.5, yti - 0.3, zs - 0.5, zs + 2.0))
    xo, xi, xw = bx - t - 0.3, bx - t - 1.5, bx - t + 0.5
    ys_lo, ys_hi = sorted(set(y for (_, y) in lid_screw_points(d)))
    for (ya, yb) in ((d['door_y'] + 6, d['bay_y1'] - 1), (d['shelf_y1'] + 1, ys_lo - 5),
                     (ys_lo + 5, ys_hi - 5), (ys_hi + 5, yti - 14)):
        for sx in (-1, 1):
            adds.append(g.box(min(sx * xi, sx * xw), max(sx * xi, sx * xw), ya, yb, zs - 3.0, zs))
            adds.append(g.box(min(sx * xi, sx * xo), max(sx * xi, sx * xo), ya, yb, zs - 0.5, zs + 2.0))
    apply_details(comp, body, adds, cuts, 'BackLid')


# ---------------------------------------------------------------- battery door
def detail_bay_door(comp, body, d):
    g = Geo()
    yb = d['body_y_bot']
    cuts = []
    for (x, z) in door_screw_points(d):
        cuts.append(g.cyl((x, yb - 1, z), (x, d['door_y'] + 0.1, z), d['m3_clear'] / 2))
        cuts.append(g.cyl((x, yb - 1, z), (x, yb + 3.2, z), d['m3_head_d'] / 2))
    for i in range(-6, 7):                                  # bank ventilation
        x = 6.0 * i
        cuts.append(g.box(x - 1.0, x + 1.0, yb - 1, yb + d['wall'] + 1, d['bay_zc'] - 15, d['bay_zc'] + 15))
    apply_details(comp, body, [], cuts, 'BayDoor')


# ---------------------------------------------------------------- wings
def detail_wing_front(comp, body, d, s):
    g = Geo()
    t, sf, sw = d['wall'], d['sf'], d['sw']
    F = slab_frame(d)
    adds, cuts = [], []
    X = lambda x: s * x
    xsc = X(tab_screw_x(d))
    # countersunk holes over the centre piece's front-tab inserts (square to the raked face)
    for yy in (-7, 7):
        cuts.append(g.ocyl(F, (xsc, yy, sf - t - 1), (xsc, yy, sf + 1), d['m3_clear'] / 2))
        cuts.append(g.ocyl(F, (xsc, yy, sf - 1.8), (xsc, yy, sf + 0.01), d['m3_clear'] / 2, d['m3_cs_d'] / 2))
    # index-finger trigger on the grip front, on a raised flat pad (a seat for the button flange and a
    # landmark the finger can find without looking)
    tx, ty = grip_xy(d, s, 0.0, d['trigger_y'])
    tyl = sy(d, ty)
    adds.append(g.ocyl(F, (tx, tyl, sf - 2.0), (tx, tyl, sf + 1.2), 8.5))
    cuts.append(g.ocyl(F, (tx, tyl, sf - t - 3), (tx, tyl, sf + 2), d['btn_d'] / 2))
    # front/back joining bosses along the slab normal, inserts at the (raked) split face
    for (x, y) in wing_screw_points(d):
        yl = sy(d, y)
        adds.append(g.ocyl(F, (X(x), yl, sw), (X(x), yl, sf - t + 0.5), d['m3_boss_d'] / 2))
        cuts.append(g.ocyl(F, (X(x), yl, sw - 0.1), (X(x), yl, sw + d['m3_insert_l']), d['m3_insert_d'] / 2))
    apply_details(comp, body, adds, cuts, 'WingFront_' + ('L' if s > 0 else 'R'))


def detail_wing_back(comp, body, d, s):
    g = Geo()
    t, sf, sw = d['wall'], d['sf'], d['sw']
    F = slab_frame(d)
    adds, cuts = [], []
    X = lambda x: s * x
    for (x, y) in wing_screw_points(d):
        yl = sy(d, y)
        adds.append(g.ocyl(F, (X(x), yl, -sf + t - 0.5), (X(x), yl, sw), d['m3_boss_d'] / 2))
        cuts.append(g.ocyl(F, (X(x), yl, -sf - 1), (X(x), yl, sw + 0.1), d['m3_clear'] / 2))
        cuts.append(g.ocyl(F, (X(x), yl, -sf - 1), (X(x), yl, -sf + 3.2), d['m3_head_d'] / 2))
    # countersunk holes over the centre piece's back-tab inserts
    for (x, y) in back_tab_screw_points(d):
        yl = sy(d, y)
        cuts.append(g.ocyl(F, (X(x), yl, -sf - 1), (X(x), yl, -sf + t + 1), d['m3_clear'] / 2))
        cuts.append(g.ocyl(F, (X(x), yl, -sf - 0.01), (X(x), yl, -sf + 1.8), d['m3_cs_d'] / 2, d['m3_clear'] / 2))
    apply_details(comp, body, adds, cuts, 'WingBack_' + ('L' if s > 0 else 'R'))


# ---------------------------------------------------------------- LED pods
def pod_frame(d, s):
    """Pod face frame: x runs outward along the face, y up, z out of the face."""
    th = math.radians(d['toe_deg'])
    o = (s * d['pod_x0'], 0.0, d['body_z_front'] + d['pod_base'])
    ex = (s * math.cos(th), 0.0, math.sin(th))
    ey = (0.0, 1.0, 0.0)
    ez = (-s * math.sin(th), 0.0, math.cos(th))
    return (o, ex, ey, ez)


def uv_frame(d, s):
    """UV sub-frame: origin on the pod face at the top-centre of the UV board area, tilted up by uv_tilt."""
    o, ex, ey, ez = pod_frame(d, s)
    a = math.radians(d['uv_tilt_deg'])
    L = d['pod_face_w']
    o2 = tuple(o[i] + (L / 2) * ex[i] + d['y_uv'] * ey[i] for i in range(3))
    ez2 = tuple(math.cos(a) * ez[i] + math.sin(a) * ey[i] for i in range(3))
    ey2 = tuple(math.cos(a) * ey[i] - math.sin(a) * ez[i] for i in range(3))
    return (o2, ex, ey2, ez2)


def uv_led_points(d):
    """(x', y') of the UV LEDs in the UV sub-frame (3 boards side by side, 3 LEDs each)."""
    pts = []
    for i in range(d['uv_n']):
        x = (i - (d['uv_n'] - 1) / 2) * d['uv_col_gap']
        for j in range(d['uv_per']):
            pts.append((x, -0.4 - (d['uv_board'][0] - (d['uv_per'] - 1) * d['uv_pitch']) / 2 - j * d['uv_pitch']))
    return pts


def uv_board_front(d):
    """z' (UV sub-frame) of the socket board's front face: skin back, 0.5 mm lead gap, socket height."""
    return -(d['uv_skin'] + 0.5 + d['uv_sock_h'])


def uv_holes(d, s):
    """UV board holes in the pod's UV frame. The same board serves both pods; in the mirror-image right pod it
    is turned round, so its hole pattern is the left one with x reversed (in the mirrored frame)."""
    return [(x if s > 0 else -x, y) for (x, y) in d['uv_pcb_holes']]


def white_module_cy(d):
    """Pod y of the white modules' centre (carrier: upper tab holes 15 mm below its top edge; gen_pcb WHITE_TOP)."""
    return d['y_wtop'] - 15.0 - d['white_hole_pitch'] / 2


def white_columns(d):
    n, w, gap = d['white_n'], d['white_w'], d['white_gap']
    lx0 = (d['pod_face_w'] - (n * w + (n - 1) * gap)) / 2
    return [lx0 + w / 2 + i * (w + gap) for i in range(n)]


def build_led_pod(comp, d, s, name):
    g = Geo()
    th = math.radians(d['toe_deg'])
    zf = d['body_z_front']
    x0, x1, y0, y1 = d['pod_x0'], d['pod_x1'], d['pod_y0'], d['pod_y1']
    L = d['pod_face_w']
    fr, uf = pod_frame(d, s), uv_frame(d, s)
    zmax = zf + d['pod_base'] + (x1 - x0) * math.tan(th) + 25
    column = g.box(min(s * x0, s * x1), max(s * x0, s * x1), y0, y1, zf, zmax)
    behind = g.obox(fr, -100, L + 100, y0 - 5, y1 + 5, -200, 0)
    pod = g.intersect(column, behind)
    # UV wedge: a tilted platform so the UV LEDs point up toward the image centre, plus a light frame
    hw, hh, fw = d['uv_cav_w'] / 2, d['uv_h'], 1.6
    wedge = g.intersect(column, g.obox(uf, -hw - fw, hw + fw, -hh - fw, fw, -60, 0))
    frame_ = g.subtract(g.obox(uf, -hw - fw, hw + fw, -hh - fw, fw, -1, d['uv_frame_h']),
                        g.obox(uf, -hw, hw, -hh, 0, -2, d['uv_frame_h'] + 1))
    # two tongues on the pod back drop into the top wire window, so the long pod cannot lift
    tongues = [g.box(min(s * xa, s * xb), max(s * xa, s * xb), d['ytop'] - 0.5, d['ytop'] + 3.3, zf - 2.0, zf + 0.5)
               for (xa, xb) in ((x0 + 4.5, x0 + 10.5), (x1 - 10.5, x1 - 4.5))]
    bosses = [g.ocyl(uf, (x, y, -d['uv_skin'] + 0.2), (x, y, uv_board_front(d)), d['uv_boss_d'] / 2)
              for (x, y) in uv_holes(d, s)]
    pod = g.union([pod, wedge, frame_] + tongues + bosses)

    cuts = []
    # WHITE carrier: one pocket (module tops flush with the face), a lip over the inner edge, 2 x M2 inserts at the
    # outer edge, a hollow behind for the tab nuts and the J4 cable, and a channel down to the UV cavity (pod hub J2)
    pw_, ph_, pt_ = d['white_pcb']
    u0, c_ = L / 2 - pw_ / 2, d['clear']
    zbk = -(d['white_mod_h'] + pt_)                                   # carrier back face
    vy = lambda v: d['y_wtop'] - v                                    # carrier v -> pod y
    cuts.append(g.obox(fr, u0 + 2.5, u0 + pw_ + c_, vy(ph_) - c_, vy(0) + c_, zbk, 1))                # open pocket
    cuts.append(g.obox(fr, u0 - c_, u0 + 2.5 + 0.01, vy(ph_) - c_, vy(0) + c_, zbk, zbk + pt_ + 0.3))   # under the lip
    cuts.append(g.obox(fr, u0 + 6, u0 + pw_ - 6, vy(ph_ - 1), vy(1), -200, zbk + 0.01))              # nuts + J4 hollow
    cuts.append(g.obox(fr, L / 2 - 3, L / 2 + 3, d['y_uv'] - 2.0, vy(ph_ - 2), -200, -5.0))          # cable channel
    for (u, v) in d['white_pcb_holes']:
        cuts.append(g.ocyl(fr, (u0 + u, vy(v), zbk + 0.1), (u0 + u, vy(v), zbk - d['m2_insert_l']), d['m2_insert_d'] / 2))
    # UV LEDs: holes through the skin with a FRONT flange counterbore (each LED pulls out from the front);
    # board cavity open to the back; the socket board sits on two M2 bosses
    for (x, y) in uv_led_points(d):
        cuts.append(g.ocyl(uf, (x, y, 1), (x, y, -d['uv_skin'] - 0.5), d['uv_led_d'] / 2))
        cuts.append(g.ocyl(uf, (x, y, 1), (x, y, -(d['uv_skin'] - 1.0)), 3.15))
    cuts.append(g.obox(uf, -hw, hw, -hh, 0, -120, -d['uv_skin']))
    zb = uv_board_front(d)
    for (x, y) in uv_holes(d, s):
        cuts.append(g.ocyl(uf, (x, y, zb + 0.1), (x, y, zb + d['m2_insert_l']), d['m2_insert_d'] / 2))
    # mounting screws (along Z, into the FrontCentre inserts), counterbored from the face
    for (x, y) in pod_screw_points(d, s):
        cuts.append(g.cyl((x, y, zf - 1), (x, y, zmax + 1), d['m3_clear'] / 2))
        cuts.append(g.cyl((x, y, zf + 3.0), (x, y, zmax + 1), d['m3_head_d'] / 2))
    pod = g.subtract(pod, g.union(cuts))
    return add_temp_body(comp, pod, name)


# ---------------------------------------------------------------- camera hood
def build_cam_hood(comp, d):
    """Removable cone hood: no vignetting at the diagonal FOV + 3 deg; screws on lugs above/below."""
    g = Geo()
    zf, cy = d['body_z_front'], d['disp_cy']
    half = math.radians(d['cam_dfov'] / 2 + 3)
    r_lens = d['cam_barrel_d'] / 2 + 0.3
    z_tip = zf + d['hood_l']
    r_in_face = r_lens + d['cam_set_back'] * math.tan(half)
    r_in_tip = r_lens + (d['cam_set_back'] + d['hood_l']) * math.tan(half)
    dy = d['hood_screw_dy']
    hood = g.union([g.cyl((0, cy, zf), (0, cy, zf + 2.0), 17.0),
                    g.box(-6, 6, cy - dy - 4.5, cy + dy + 4.5, zf, zf + 2.0),
                    g.cyl((0, cy, zf), (0, cy, z_tip), r_in_face + 3.0, r_in_tip + 1.6)])
    cuts = [g.cyl((0, cy, zf - 0.1), (0, cy, z_tip + 0.1), r_in_face, r_in_tip)]
    for sy in (-1, 1):
        cuts.append(g.cyl((0, cy + sy * dy, zf - 0.1), (0, cy + sy * dy, zf + 2.1), d['m2_clear'] / 2))
    return add_temp_body(comp, g.subtract(hood, g.union(cuts)), 'CamHood')


# ---------------------------------------------------------------- inspection hood (device v2)
def hood_geom(d):
    """Hood levels and its 4 walls. Each wall: (axis, s, u_top, u_bot) with u measured outward
    (u = s * coordinate) at the top (z_top, collar front) and at the rim (z_bot, plate plane)."""
    c, t = d['clear'], d['wall']
    cy = d['disp_cy']
    zt = d['body_z_front']
    zb = d['body_z_front'] - d['cam_set_back'] + d['hood_wd']
    bw, bh = d['hood_bottom']
    ax = d['bx'] + c + t
    walls = [('x', 1, ax, bw / 2), ('x', -1, ax, bw / 2),
             ('y', 1, d['body_y_top'] + c + t, cy + bh / 2),
             ('y', -1, -(d['body_y_bot'] - c - t), -(cy - bh / 2))]
    return dict(zt=zt, zb=zb, cy=cy, walls=walls, t=t)


def wall_u(w, hg, z, off=0.0):
    """Outward coordinate of a wall's outer face at height z, moved inward by off."""
    _, _, ut, ub = w
    return ut + (ub - ut) * (z - hg['zt']) / (hg['zb'] - hg['zt']) - off


def wall_frame(w, hg, off=0.0):
    """Frame on a wall plane (moved inward by off): ex = outward normal, ey = along the wall
    horizontally, ez = down the wall (toward the plate)."""
    axis, s, ut, ub = w
    du, dz = ub - ut, hg['zb'] - hg['zt']
    L = math.hypot(du, dz)
    n_u, n_z = dz / L, -du / L
    e_u, e_z = du / L, dz / L
    if axis == 'x':
        n, e, o_h, along = (s * n_u, 0.0, n_z), (s * e_u, 0.0, e_z), (s * (ut - off), 0.0, hg['zt']), (0.0, 1.0, 0.0)
    else:
        n, e, o_h, along = (0.0, s * n_u, n_z), (0.0, s * e_u, e_z), (0.0, s * (ut - off), hg['zt']), (1.0, 0.0, 0.0)
    return (o_h, n, along, e)


def frustum_region(g, hg, off, z0, z1):
    """Solid inside the 4 hood wall planes moved inward by off, between z0 and z1."""
    slab = g.box(-400, 400, -400, 400, z0, z1)
    outside = g.union([g.obox(wall_frame(w, hg, off), 0, 600, -600, 600, -600, 600) for w in hg['walls']])
    return g.subtract(slab, outside)


def rrect(g, hw, y0, y1, rt, rb, z0, z1):
    """Prism along z of a rectangle |x| <= hw, y0..y1, with top corner radius rt and bottom rb."""
    parts = [g.box(-hw, hw, y0 + rb, y1 - rt, z0, z1),
             g.box(-hw + rt, hw - rt, y1 - rt, y1, z0, z1),
             g.box(-hw + rb, hw - rb, y0, y0 + rb, z0, z1)]
    for sx in (-1, 1):
        parts.append(g.cyl((sx * (hw - rt), y1 - rt, z0), (sx * (hw - rt), y1 - rt, z1), rt))
        parts.append(g.cyl((sx * (hw - rb), y0 + rb, z0), (sx * (hw - rb), y0 + rb, z1), rb))
    return g.union(parts)


def ring_frames(d, hg):
    """Frame per ring LED module: origin at the board face centre, ex along the module (parallel to its
    wall), ey across it, ez its emission axis: square to the wall in plan and tilted down at the plate's
    centre line (a 78 mm bar of 150 deg LEDs needs no swing in plan, and a swing drives one end into the wall)."""
    out = []
    z = hg['zb'] - d['ring_h']
    for axis, s, along in d['ring_mods']:
        w = next(w for w in hg['walls'] if w[0] == axis and w[1] == s)
        u = wall_u(w, hg, z, hg['t'] + d['ring_inset'])
        o = (s * u, hg['cy'] + along, z) if axis == 'x' else (along, s * u, z)
        target = (0.0, o[1], hg['zb']) if axis == 'x' else (o[0], hg['cy'], hg['zb'])
        ez = [target[i] - o[i] for i in range(3)]
        m = math.sqrt(sum(v * v for v in ez)); ez = [v / m for v in ez]
        hz = (0.0, 1.0, 0.0) if axis == 'x' else (1.0, 0.0, 0.0)
        dot = sum(hz[i] * ez[i] for i in range(3))
        ex = [hz[i] - dot * ez[i] for i in range(3)]
        m = math.sqrt(sum(v * v for v in ex)); ex = [v / m for v in ex]
        ey = [ez[1] * ex[2] - ez[2] * ex[1], ez[2] * ex[0] - ez[0] * ex[2], ez[0] * ex[1] - ez[1] * ex[0]]
        out.append(((o, tuple(ex), tuple(ey), tuple(ez)), w, along))
    return out


def lip_frame(d, hg, w, along):
    """45 deg lip from the wall, above a module: ex along the wall, ez inward and toward the camera."""
    axis, s = w[0], w[1]
    z = hg['zb'] - d['ring_h'] - d['white_w'] / 2 - d['lip_gap']
    u = wall_u(w, hg, z, hg['t'])
    o = (s * u, hg['cy'] + along, z) if axis == 'x' else (along, s * u, z)
    c = math.sqrt(0.5)
    ez = (-s * c, 0.0, -c) if axis == 'x' else (0.0, -s * c, -c)
    ex = (0.0, 1.0, 0.0) if axis == 'x' else (1.0, 0.0, 0.0)
    ey = (ez[1] * ex[2] - ez[2] * ex[1], ez[2] * ex[0] - ez[0] * ex[2], ez[0] * ex[1] - ez[1] * ex[0])
    return (o, ex, ey, ez)


def build_inspection_hood(comp, d):
    """Light-tight hood: a collar that slides over the body front, a matte-black frustum down to the
    plate with a thickened rim and a foam-seal groove, 6 angled LED-module pockets 18 mm above the plate
    with a 45 deg lip over each (the camera must not see the LEDs), thumb-screw holes and wire slots.
    Print rim-down in black PETG; tree supports under the 6 module brackets (the lips need none)."""
    g = Geo()
    hg = hood_geom(d)
    zt, zb, t, c = hg['zt'], hg['zb'], hg['t'], d['clear']
    z0 = d['hood_collar_z0']
    bx = d['bx']
    yb, yt = d['body_y_bot'], d['body_y_top']
    rt, rb = d['body_r_top'], d['body_r_bot']
    collar = g.subtract(rrect(g, bx + c + t, yb - c - t, yt + c + t, rt + c + t, rb + c + t, z0, zt + 0.01),
                        rrect(g, bx + c, yb - c, yt + c, rt + c, rb + c, z0 - 1, zt + 1))
    outer = frustum_region(g, hg, 0.0, zt, zb)
    cavity = frustum_region(g, hg, t, zt - 1, zb + 1)
    rw, rh = d['hood_rim']
    rim = g.subtract(frustum_region(g, hg, t, zb - rh, zb), frustum_region(g, hg, t + rw, zb - rh - 1, zb + 1))
    hood = g.union([collar, g.subtract(outer, cavity), rim])
    adds = []
    for fr, w, along in ring_frames(d, hg):
        L, W = d['white_hole_pitch'] / 2 + 3.0 + c + 2.0, d['white_w'] / 2 + c + 2.0
        adds.append(g.obox(fr, -L, L, -W, W, -(d['ring_inset'] + 14.0), 0.0))
        adds.append(g.obox(lip_frame(d, hg, w, along), -L - 1.0, L + 1.0, -d['lip_t'] / 2, d['lip_t'] / 2, -4.0, d['lip_len']))
    outer_all = g.union([frustum_region(g, hg, 0.0, zt, zb),
                         rrect(g, bx + c + t, yb - c - t, yt + c + t, rt + c + t, rb + c + t, z0, zt + 0.01)])
    hood = g.union([hood, g.intersect(g.union(adds), outer_all)])
    cuts = []
    for fr, w, along in ring_frames(d, hg):
        L, W = d['white_hole_pitch'] / 2 + 3.0 + c, d['white_w'] / 2 + c
        cuts.append(g.obox(fr, -L, L, -W, W, -d['white_pocket'], 3.0))           # module pocket, tabs included
        for hx in (-d['white_hole_pitch'] / 2, d['white_hole_pitch'] / 2):        # M2 x 8 through the tabs into inserts
            cuts.append(g.ocyl(fr, (hx, 0, -d['white_pocket'] + 0.1), (hx, 0, -d['white_pocket'] - d['m2_insert_l']), d['m2_insert_d'] / 2))
        cuts.append(g.obox(fr, L - 6.0, L + 1.5, -3.0, 3.0, -(d['ring_inset'] - 1.5), 1.0))   # wire exit; stops short of the wall (no light path)
    sw, sd = d['hood_seal']
    mid = t + rw / 2
    cuts.append(g.subtract(frustum_region(g, hg, mid - sw / 2, zb - sd, zb + 1),
                           frustum_region(g, hg, mid + sw / 2, zb - sd - 1, zb + 2)))   # foam groove
    sy, sz = d['hood_screw']
    wy, wz, wl, wh = d['hood_wire']
    for s in (-1, 1):
        x_in, x_out = s * (bx + c - 0.5), s * (bx + c + t + 4.0)
        cuts.append(g.cyl((x_in, sy, sz), (x_out, sy, sz), d['m3_clear'] / 2))
        cuts.append(g.box(min(s * (bx - 1), s * (bx + c + t + 1)), max(s * (bx - 1), s * (bx + c + t + 1)),
                          wy - wl / 2, wy + wl / 2, wz - wh / 2, wz + wh / 2))
    hood = g.subtract(hood, g.union(cuts))
    pads = []
    for s in (-1, 1):   # thumb-screw pads: 3 mm extra thread length
        pads.append(g.subtract(g.cyl((s * (bx + c + t - 0.3), sy, sz), (s * (bx + c + t + 3.0), sy, sz), 6.0),
                               g.cyl((s * (bx + c), sy, sz), (s * (bx + c + t + 4.0), sy, sz), d['m3_clear'] / 2)))
    hood = g.union([hood, g.union(pads)])
    return add_temp_body(comp, hood, 'InspectionHood')


def hood_mount_front_centre(comp, body, d):
    """Heat-set insert bosses in the FrontCentre side walls for the hood's two thumb screws."""
    g = Geo()
    sy, sz = d['hood_screw']
    bx, t = d['bx'], d['wall']
    adds, cuts = [], []
    for s in (-1, 1):
        adds.append(g.cyl((s * (bx - t + 0.2), sy, sz), (s * (bx - t - d['m3_insert_l'] - 1.0), sy, sz), d['m3_boss_d'] / 2))
        cuts.append(g.cyl((s * (bx + 1.0), sy, sz), (s * (bx - d['m3_insert_l'] - 0.6), sy, sz), d['m3_insert_d'] / 2))
    apply_details(comp, body, adds, cuts, 'hood_mount')


def build_hood_reference(comp, d):
    """Fit-check dummies: a 100 x 100 x 5 plate at the working distance and the 6 ring LED modules."""
    g = Geo()
    hg = hood_geom(d)
    cy, zb = hg['cy'], hg['zb']
    add_temp_body(comp, g.box(-50, 50, cy - 50, cy + 50, zb, zb + 5), 'Plate100')
    mods = [g.obox(fr, -d['white_l'] / 2, d['white_l'] / 2, -d['white_w'] / 2, d['white_w'] / 2, -d['white_pocket'] + 0.05,
                   -d['white_pocket'] + d['white_mod_h'])
            for fr, _, _ in ring_frames(d, hg)]
    add_temp_body(comp, g.union(mods), 'RingLeds')


# ---------------------------------------------------------------- board placements (KiCad STEP -> model)
def _frame_pt(fr, x, y, z):
    o, ex, ey, ez = fr
    return tuple(o[i] + x * ex[i] + y * ey[i] + z * ez[i] for i in range(3))


def _neg(v):
    return tuple(-c for c in v)


def assembly_frames(d):
    """Where each KiCad board sits: {board: (origin, X, Y, Z)} in model mm. The STEP is exported with the
    board corner (u = 0, v = 0) as origin (kicad-cli --user-origin 100x100mm): STEP X = u, Y = -v, and
    Z = the component (F) side normal, with the board's back face at Z = 0. Mirror-image pod frames use a
    right-handed frame with X reversed (the R carrier is the mirrored board; the UV board is turned round)."""
    out = {}
    yb = d['shelf_y1'] + d['pcb_standoff']
    out['power'] = ((21.0, yb, -32.0), (1, 0, 0), (0, 0, -1), (0, 1, 0))
    out['light'] = ((-21.0, yb, 23.0), (-1, 0, 0), (0, 0, 1), (0, 1, 0))
    xw = d['bx'] - d['wall'] - 0.2
    out['charge'] = ((xw, d['chg_y'], d['chg_z0']), (0, 0, 1), (1, 0, 0), (0, 1, 0))
    pw_, ph_, pt_ = d['white_pcb']
    L = d['pod_face_w']
    zbk = -(d['white_mod_h'] + pt_)
    zb = uv_board_front(d)
    for side, s in (('l', 1), ('r', -1)):
        fr, uf = pod_frame(d, s), uv_frame(d, s)
        u0 = L / 2 - pw_ / 2
        if s > 0:
            out['white_l'] = (_frame_pt(fr, u0, d['y_wtop'], zbk), fr[1], fr[2], fr[3])
            out['uv_l'] = (_frame_pt(uf, -21.5, -0.4, zb - d['uv_pcb'][2]), uf[1], uf[2], uf[3])
        else:
            out['white_r'] = (_frame_pt(fr, u0 + pw_, d['y_wtop'], zbk), _neg(fr[1]), fr[2], fr[3])
            out['uv_r'] = (_frame_pt(uf, 21.5, -0.4, zb - d['uv_pcb'][2]), _neg(uf[1]), uf[2], uf[3])
        # pod dock: behind the UV window in the front wall, ZIF facing the window (and the pod's flex)
        xa, xb = d['pod_x0'] + 4, d['pod_x1'] - 4
        xc, yc = s * (xa + xb) / 2, (d['pod_y0'] + 2 + d['y_uv'] + 0.5) / 2
        out['pod_dock_' + side] = ((xc - 14.0, yc + 7.0, d['zf_in'] - 1.8), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    # hood dock under the top wall near the front, spring pins up; hood pad face down on them
    zc = d['body_z_front'] - 7.0
    out['hood_dock'] = ((-17.0, d['y_top_in'] - 1.8, zc - 6.5), (1, 0, 0), (0, 0, -1), (0, 1, 0))
    out['hood_pad'] = ((-20.0, d['body_y_top'] + 1.9, zc + 0.3), (1, 0, 0), (0, 0, 1), (0, -1, 0))
    # button boards inside each wing's arm, parallel to the slab, between the thumb buttons and the trigger
    F = slab_frame(d)
    for side, s in (('l', 1), ('r', -1)):
        xc = s * d['btn_pcb_x']
        out['btn_' + side] = (_frame_pt(F, xc - 18.0, sy(d, d['btn_pcb_y']) + 10.0, d['btn_pcb_z']), F[1], F[2], F[3])
    return out


def button_dummies(d, g):
    """Panel buttons: index triggers (12 mm) on the grip fronts, thumbs (7 mm) on the arm-root backs."""
    F = slab_frame(d)
    sf, t = d['sf'], d['wall']
    out = {}
    for side, s in (('L', 1), ('R', -1)):
        tx, ty = grip_xy(d, s, 0.0, d['trigger_y'])
        tyl = sy(d, ty)
        out['Trigger_' + side] = g.union([g.ocyl(F, (tx, tyl, sf + 1.2), (tx, tyl, sf + 4.0), 5.5),        # cap
                                          g.ocyl(F, (tx, tyl, sf - t - 16.0), (tx, tyl, sf + 1.2), 5.9)])  # body
        thumbs = []
        for (x, y) in thumb_button_points(d):
            yl = sy(d, y)
            thumbs.append(g.ocyl(F, (s * x, yl, -sf - 2.0), (s * x, yl, -sf), 3.2))
            thumbs.append(g.ocyl(F, (s * x, yl, -sf), (s * x, yl, -sf + t + 9.0), 3.4))
        out['Thumbs_' + side] = g.union(thumbs)
    out['PwrCap'] = g.cyl((d['pwr_btn_x'], d['y_top_in'], d['bay_zc']), (d['pwr_btn_x'], d['body_y_top'] + 1.5, d['bay_zc']), 6.0)
    return out


# ---------------------------------------------------------------- reference dummies
def build_reference(comp, d):
    g = Geo()
    cy = d['disp_cy']
    items = {}
    gw, gh = d['disp_glass']
    pw, ph = d['disp_pcb']
    pox, poy = d['disp_pcb_off']
    rw, rh, rz = d['disp_parts']
    zg = d['z_glass']
    disp = [g.box(-gw / 2, gw / 2, cy - gh / 2, cy + gh / 2, zg, zg + 2.0),
            g.box(-pw / 2 + pox, pw / 2 + pox, cy + poy - ph / 2, cy + poy + ph / 2, zg + 2.0, d['z_disp_back']),
            g.box(-rw / 2 + pox, rw / 2 + pox, cy + poy - rh / 2, cy + poy + rh / 2, d['z_disp_back'], d['z_disp_back'] + rz)]
    if d['disp_zif']:
        x0, x1, y0, y1, zmax = d['disp_zif']
        disp.append(g.box(x0, x1, cy + y0, cy + y1, d['z_disp_back'], zg + zmax))
    if d['disp_tab']:
        tx0, tx1, tdep = d['disp_tab']
        disp.append(g.box(tx0, tx1, cy - gh / 2 - tdep + 0.3, cy - gh / 2, zg, zg + 2.0))
    for (x, y) in display_post_points(d):                       # the standoffs the posts press on
        disp.append(g.cyl((x, y, d['z_disp_back']), (x, y, zg + d['disp_post_z']), 2.75))
    items['Display'] = g.union(disp)
    z0, z1 = d['z_pcb0'], d['z_pcb1']
    zh0, zh1 = d['z_hat0'], d['z_hat1']
    B = lambda xa, xb, ya, yb, za, zb: pi_box(g, d, xa, xb, ya, yb, za, zb)
    pi = [B(0, 85, 0, 56, z0, z1),                              # Pi PCB
          B(2, 83, 2, 54, z0 - 3.3, z0),                        # underside parts/pegs
          B(66.45, 88.05, 2.28, 18.21, z1, z1 + 14.0),          # RJ45
          B(70.4, 88.0, 21.75, 36.25, z1, z1 + 16.5),           # USB 3.0
          B(70.3, 87.8, 39.75, 54.25, z1, z1 + 16.5),           # USB 2.0
          B(6.73, 15.67, -1.2, 8, z1, z1 + 3.3),                # USB-C
          B(22.2, 42.8, -1.63, 8, z1, z1 + 3.4),                # micro-HDMI x2
          B(7.1, 57.9, 49.96, 55.04, z1, zh0),                  # GPIO header + stacking header
          B(6.4, 64.2, 9.5, 49.6, z1 + 2.8, z1 + 13.8),         # Active Cooler incl. fan screw heads
          B(0, 65, 0, 56.5, zh0, zh1),                          # AI HAT+ 2 PCB
          B(3.6, 47.45, 6.5, 49.0, zh1, zh1 + 6.0),             # HAT heatsink (main)
          B(47.45, 64.65, 16.2, 49.0, zh1, zh1 + 6.0),          # HAT heatsink (beside the FFC notch)
          B(7.4, 57.6, 50.0, 55.0, zh1, zh1 + 8.5)]             # J1 pass-through socket on the HAT
    for (px, py) in ((3.5, 9.5), (61.5, 46.5)):                 # heatsink push-pin heads
        X, Y = pi_xy(d, px, py)
        pi.append(g.cyl((X, Y, zh1), (X, Y, z1 + d['stack_top']), 2.8))
    for (px, py) in ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)):   # HAT screw heads
        X, Y = pi_xy(d, px, py)
        pi.append(g.cyl((X, Y, zh1), (X, Y, zh1 + 1.7), 2.25))
    items['PiStack'] = g.union(pi)
    items['FFC_keepout'] = B(47.1, 57.5, 0.2, 16.0, z1, zh1 + 10.0)
    cam_x0 = -d['cam_w'] / 2
    items['Camera'] = g.union([
        g.box(cam_x0, -cam_x0, d['cam_top'] - d['cam_h'], d['cam_top'], d['z_cam_front'] - d['cam_pcb_t'], d['z_cam_front']),
        g.box(-6.0, 6.0, cy - 6.25, cy + 6.25, d['z_cam_front'], d['z_cam_front'] + 4.9),
        g.cyl((0, cy, d['z_cam_front']), (0, cy, d['z_lens']), d['cam_barrel_d'] / 2),
        g.box(-9.8, 9.8, d['cam_top'] - d['cam_h'], d['cam_top'] - d['cam_h'] + 5.7,
              d['z_cam_front'] - d['cam_pcb_t'] - 2.75, d['z_cam_front'] - d['cam_pcb_t'])])
    bl, bw, bt = d['bank']
    items['PowerBank'] = g.box(d['bay_x0'] + 1, d['bay_x0'] + 1 + bl, d['bay_y0'] + 0.3, d['bay_y0'] + 0.3 + bt,
                               d['bay_zc'] - bw / 2, d['bay_zc'] + bw / 2)
    fs, ft = d['fan']
    cw, ch_, ct = d['chg_pcb']                                    # CHARGE board + receptacle + JST
    xw = d['bx'] - d['wall'] - 0.2
    items['ChargeBoard'] = g.union([
        g.box(xw - ch_, xw, d['chg_y'], d['chg_y'] + ct, d['chg_z0'], d['chg_z0'] + cw),
        g.box(xw - 7.6, xw + 0.2, d['chg_y'] + ct, d['chg_y'] + ct + 3.3, d['chg_z0'] + 1.5, d['chg_z0'] + 10.5),
        g.box(xw - 11.0, xw - 1.0, d['chg_y'] + ct, d['chg_y'] + ct + 6.0, d['chg_z0'] + 13.5, d['chg_z0'] + 18.5)])
    items['PwrButton'] = g.cyl((d['pwr_btn_x'], d['y_top_in'] - 22.0, d['bay_zc']), (d['pwr_btn_x'], d['y_top_in'], d['bay_zc']), 7.5)
    items['UsbExport'] = g.box(d['usb_a_x'] - 7.0, d['usb_a_x'] + 7.0, d['y_top_in'] - 12.0, d['y_top_in'],
                               d['bay_zc'] - 10.0, d['bay_zc'] + 10.0)
    items.update(button_dummies(d, g))
    items['Fan30'] = g.box(d['fan_cx'] - fs / 2, d['fan_cx'] + fs / 2, d['y_top_in'] - 6 - ft, d['y_top_in'] - 6,
                           d['fan_cz'] - fs / 2, d['fan_cz'] + fs / 2)
    yb = d['shelf_y1'] + d['pcb_standoff']              # board underside
    for name, s in (('PcbPower', 1), ('PcbLight', -1)):
        x0, x1, z0, z1 = pcb_main_box(d, s)
        la, lb, lh = d['pcb_low']
        lo0, lo1 = (la, lb) if s > 0 else (-lb, -la)
        yt = yb + d['pcb_t']
        tall = g.subtract(g.box(x0 + 1, x1 - 1, yt, yt + d['pcb_top'], z0 + 1, z1 - 1),
                          g.box(lo0, lo1, yt + lh, yt + d['pcb_top'] + 1, z0, z1))      # low-profile strip
        items[name] = g.union([g.box(x0, x1, yb, yt, z0, z1),                              # board
                               tall,
                               g.box(x0 + 7, x1 - 7, yb - d['pcb_bot'], yb, z0 + 7, z1 - 7)])   # underside parts, clear of the bosses
    for side, s in (('L', 1), ('R', -1)):
        fr, uf = pod_frame(d, s), uv_frame(d, s)
        pw_, ph_, pt_ = d['white_pcb']
        L = d['pod_face_w']
        zbk = -(d['white_mod_h'] + pt_)
        vc = white_module_cy(d)                                              # module centre
        leds = [g.obox(fr, L / 2 - pw_ / 2, L / 2 + pw_ / 2, d['y_wb'], d['y_wtop'], zbk, zbk + pt_)]   # WHITE carrier
        leds += [g.obox(fr, cx - d['white_w'] / 2, cx + d['white_w'] / 2, vc - d['white_l'] / 2, vc + d['white_l'] / 2,
                        zbk + pt_, zbk + pt_ + d['white_mod_h'] - 0.05) for cx in white_columns(d)]          # modules
        leds += [g.obox(fr, cx - 3.6, cx + 3.6, d['y_wtop'] - 11.2, d['y_wtop'] - 1.2, zbk + pt_, zbk + pt_ + 5.0)
                 for cx in white_columns(d)]                                                                # push-in terminals
        leds.append(g.obox(fr, L / 2 + 4, L / 2 + 10, vc - 9, vc + 8.5, zbk - 5.0, zbk))                    # J4 side-entry + plug, back
        zb = uv_board_front(d)
        pw, ph, pt = d['uv_pcb']
        for (x, y) in uv_led_points(d):
            leds.append(g.ocyl(uf, (x, y, -(d['uv_skin'] - 1.0)), (x, y, 8.7 - (d['uv_skin'] - 1.0)), 2.5))   # LED, flange in front
            leds.append(g.obox(uf, x - 2.54, x + 2.54, y - 1.27, y + 1.27, zb, zb + d['uv_sock_h']))          # its socket
        leds.append(g.obox(uf, -pw / 2, pw / 2, -0.4 - ph, -0.4, zb - pt, zb))                                 # socket board
        leds.append(g.obox(uf, s * 8.0 - 2.4, s * 8.0 + 2.4, -0.4 - 9.5 - 7.3, -0.4 - 9.5 + 7.3, zb - pt - 2.1, zb - pt))   # ZIF on the back
        items['Leds_' + side] = g.union(leds)
    for k, tb in items.items():
        add_temp_body(comp, tb, k)


# ---------------------------------------------------------------- checks, view, export
def interference_report(des, enc, ref):
    bodies = [b for b in enc.bRepBodies] + [b for b in ref.bRepBodies]
    ii = des.createInterferenceInput(oc(bodies))
    ii.areCoincidentFacesIncluded = False
    res = des.analyzeInterference(ii)
    ref_names = set(b.name for b in ref.bRepBodies)
    lines = ['interference (cm3):']
    for i in range(res.count):
        r = res.item(i)
        if r.entityOne.name in ref_names and r.entityTwo.name in ref_names:
            continue                                   # dummies may overlap each other
        vol = r.interferenceBody.volume if r.interferenceBody else 0.0
        if vol > 0.0005:
            lines.append('  %s x %s : %.4f' % (r.entityOne.name, r.entityTwo.name, vol))
    if len(lines) == 1:
        lines.append('  none')
    return '\n'.join(lines)


def set_view(app, eye=(30, 18, 70)):
    vp = app.activeViewport
    cam = vp.camera
    cam.isSmoothTransition = False
    cam.target = adsk.core.Point3D.create(0, 0, 0)
    cam.eye = adsk.core.Point3D.create(*eye)
    cam.upVector = adsk.core.Vector3D.create(0, 1, 0)
    vp.camera = cam
    vp.fit()


def export_all(des, enc, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    em = des.exportManager
    for b in enc.bRepBodies:
        if b.name not in PRINTED:
            continue
        opts = em.createSTLExportOptions(b, os.path.join(out_dir, b.name + '.stl'))
        opts.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
        opts.isBinaryFormat = True
        em.execute(opts)
    em.execute(em.createSTEPExportOptions(os.path.join(out_dir, 'Ayas_handheld.step'), des.rootComponent))
    em.execute(em.createFusionArchiveExportOptions(os.path.join(out_dir, 'Ayas_handheld.f3d'), des.rootComponent))
    print('exported to', out_dir)
