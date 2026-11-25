"""TriNetra demo video: Blender shots from the Fusion CAD (run: blender -b -P blender_shots.py -- <shot> <out> [still_frame]).
Shots: title (orbit on the plate), explode (exploded view + label anchors JSON), burst (white LED 5-frame burst, open mode),
uv (UV burst), surface (3D height map of the measured dent)."""
import bpy, json, math, os, sys
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
SHOT, OUT = argv[0], argv[1]
STILL = int(argv[2]) if len(argv) > 2 else None
V = os.path.dirname(os.path.abspath(__file__))
CAD, A = os.path.join(V, "cad"), os.path.join(V, "assets")
FPS = 30
DUR = {"title": 8.5, "explode": 21.4, "burst": 14.4, "uv": 8.0, "surface": 15.1}[SHOT]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = FPS
sc.frame_start, sc.frame_end = 1, int(round(DUR * FPS))
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
eng = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
try:
    sc.eevee.taa_render_samples = 16
except Exception:
    pass
sc.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
sc.view_settings.look = "None"
sc.view_settings.exposure = -0.3

world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
bg = world.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.012, 0.014, 0.018, 1); bg.inputs[1].default_value = 1.0


def mat(name, color, rough=0.5, metal=0.0, emit=None, estr=0.0, img=None, alpha=1.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if img:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(img)
        nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
        if emit == "img":
            nt.links.new(t.outputs["Color"], b.inputs["Emission Color"]); b.inputs["Emission Strength"].default_value = estr
    if emit is not None and emit != "img":
        b.inputs["Emission Color"].default_value = (*emit, 1); b.inputs["Emission Strength"].default_value = estr
    return m


def emit_socket(m):
    return m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]


def emit_color(m):
    return m.node_tree.nodes["Principled BSDF"].inputs["Emission Color"]


M = {
    "body": mat("body", (0.045, 0.05, 0.058), 0.45),
    "shell": mat("shell", (0.05, 0.053, 0.058), 0.55),         # graphite glass-filled nylon shell
    "accent": mat("accent", (0.80, 0.13, 0.008), 0.5),          # safety-orange side wings
    "grip": mat("grip", (0.018, 0.018, 0.02), 0.85),           # charcoal soft-touch rubber grips
    "alu": mat("alu", (0.22, 0.23, 0.25), 0.4, 1.0),           # anodised aluminium LED pods
    "pod": mat("pod", (0.16, 0.17, 0.19), 0.4),
    "hood": mat("hood", (0.022, 0.024, 0.028), 0.7),
    "pcb": mat("pcb", (0.01, 0.22, 0.07), 0.45),
    "hat": mat("hat", (0.6, 0.62, 0.66), 0.25, 0.8),
    "cam": mat("cam", (0.03, 0.12, 0.06), 0.5),
    "bank": mat("bank", (0.03, 0.035, 0.04), 0.6),
    "dark": mat("dark", (0.02, 0.02, 0.025), 0.4),
    "carrier": mat("carrier", (0.05, 0.12, 0.3), 0.5),
    "led": mat("led", (0.9, 0.9, 0.9), 0.3, emit=(1.0, 0.97, 0.9), estr=0.0),
    "ledR": mat("ledR", (0.9, 0.9, 0.9), 0.3, emit=(1.0, 0.97, 0.9), estr=0.0),
    "ring": mat("ring", (0.9, 0.9, 0.9), 0.3, emit=(1.0, 0.97, 0.9), estr=0.0),
    "disp": mat("disp", (0.02, 0.02, 0.02), 0.2),
}
PARTS = {  # body -> material, exploded offset (Blender metres, after the 180 deg turn: +z = toward the operator)
    "BackLid": ("shell", (0, 0, 0.16)), "Display": ("disp", (0, 0, 0.115)), "PiStack": ("pcb", (0, 0, 0.075)),
    "Fan30": ("dark", (0, 0.02, 0.095)), "ElecCarriers": ("carrier", (0, 0.03, 0.045)), "FrontCentre": ("shell", (0, 0, 0)),
    "BayDoor": ("shell", (0, 0.15, 0)), "PowerBank": ("bank", (0, 0.10, 0.02)),
    "WingBack_L": ("accent", (0.09, 0, 0.04)), "WingBack_R": ("accent", (-0.09, 0, 0.04)),
    "WingFront_L": ("grip", (0.07, 0, -0.01)), "WingFront_R": ("grip", (-0.07, 0, -0.01)),
    "Camera": ("cam", (0, 0, -0.04)), "CamHood": ("dark", (0, 0, -0.07)),
    "LedPod_L": ("alu", (0.03, 0, -0.07)), "LedPod_R": ("alu", (-0.03, 0, -0.07)),
    "Leds_L": ("led", (0.03, 0, -0.10)), "Leds_R": ("ledR", (-0.03, 0, -0.10)),
    "InspectionHood": ("hood", (0, 0, -0.17)), "RingLeds": ("ring", (0, 0, -0.17)),
}
LABELS = {"Display": "5\" touch display", "PiStack": "Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H)", "Camera": "Raspberry Pi AI Camera",
          "Leds_L": "LED pod: white + UV", "InspectionHood": "Light-tight hood + low-angle LED ring", "PowerBank": "USB-C power bank",
          "BackLid": "Rear lid", "WingFront_R": "Grips with triggers"}

dev = bpy.data.objects.new("device", None); sc.collection.objects.link(dev)
dev.rotation_euler = (math.pi, 0, 0)
objs = {}
if SHOT != "surface":
    for name, (mk, off) in PARTS.items():
        f = os.path.join(CAD, name + ".stl")
        if not os.path.exists(f):
            continue
        bpy.ops.wm.stl_import(filepath=f, global_scale=0.001)
        o = bpy.context.selected_objects[0]; o.name = name
        o.data.materials.clear(); o.data.materials.append(M[mk])
        o.parent = dev
        bpy.ops.object.shade_auto_smooth() if hasattr(bpy.ops.object, "shade_auto_smooth") else None
        objs[name] = o
    # the station UI on the display glass (visible area 109 x 65.8 mm, centre y = 16.6 mm, glass at z = -38.8 mm)
    bpy.ops.mesh.primitive_plane_add(size=1)
    ui = bpy.context.active_object; ui.name = "UI"
    ui.rotation_euler = (math.pi, 0, 0)
    ui.data.materials.append(mat("ui", (1, 1, 1), 0.2, emit="img", estr=1.6, img=os.path.join(A, "ui_dent3d.jpg")))
    ui.parent = objs["Display"]            # the CAD parts are in mm (object scale 0.001): child units are mm
    ui.location = (0, 16.6, -38.9); ui.scale = (109.0, 65.8, 1.0)
    # AYAS wordmark on the lid strip in front of the screen (lid top at z = -40 mm, strip y 53..68 mm)
    bpy.ops.object.text_add()
    tx = bpy.context.active_object; tx.name = "Logo"
    tx.data.body = "AYAS"; tx.data.align_x = "CENTER"; tx.data.align_y = "CENTER"
    tx.data.font = bpy.data.fonts.load("/System/Library/Fonts/Avenir Next.ttc")
    tx.data.size = 11.0; tx.data.extrude = 0.4; tx.data.space_character = 1.25
    tx.data.materials.append(M["accent"])
    tx.parent = objs["BackLid"]; tx.location = (0, 60.6, -40.4); tx.rotation_euler = (math.pi, 0, 0)

# plate under the device (top face at the hood rim, z = -0.233 m) and a studio floor
PLATE_Z = -0.233
def plate(img, size=0.26):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, -0.014, PLATE_Z))
    p = bpy.context.active_object; p.name = "plate"
    p.data.materials.append(mat("plate", (0.5, 0.5, 0.5), 0.35, 0.6, img=img))
    return p
if SHOT in ("title", "burst", "uv"):
    tex = os.path.join(V, "plate_tex.png")
    pl = plate(tex)
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, PLATE_Z - 0.006))
    fl = bpy.context.active_object; fl.data.materials.append(mat("floor", (0.03, 0.032, 0.036), 0.6))

# lights
def area(name, loc, rot, size, energy, color=(1, 1, 1)):
    l = bpy.data.lights.new(name, "AREA"); l.size = size; l.energy = energy; l.color = color
    o = bpy.data.objects.new(name, l); o.location = loc; o.rotation_euler = rot; sc.collection.objects.link(o)
    return o
key = area("key", (0.5, -0.6, 0.6), (math.radians(55), 0, math.radians(40)), 0.8, 28)
rim = area("rim", (-0.6, 0.5, 0.3), (math.radians(70), 0, math.radians(-135)), 0.6, 18, (0.7, 0.8, 1.0))
top = area("top", (0, 0, 0.7), (0, 0, 0), 1.0, 10)

# camera on an orbiting rig
rig = bpy.data.objects.new("rig", None); sc.collection.objects.link(rig)
camd = bpy.data.cameras.new("cam"); camd.lens = 50
cam = bpy.data.objects.new("cam", camd); sc.collection.objects.link(cam); sc.camera = cam
cam.parent = rig
track = cam.constraints.new("TRACK_TO"); tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt)
track.target = tgt; track.track_axis = "TRACK_NEGATIVE_Z"; track.up_axis = "UP_Y"
F = sc.frame_end


def key_rot(o, frame, z_deg):
    o.rotation_euler = (0, 0, math.radians(z_deg)); o.keyframe_insert("rotation_euler", frame=frame)


anchors = {}
if SHOT == "title":
    tgt.location = (0, -0.01, -0.11); cam.location = (0, -0.95, 0.42)
    key_rot(rig, 1, -40); key_rot(rig, F, 30)
elif SHOT == "explode":
    tgt.location = (0, 0.0, -0.06); cam.location = (0, -1.05, 0.5); camd.lens = 45
    key_rot(rig, 1, -30); key_rot(rig, F, 25)
    t0, t1 = int(2.5 * FPS), int(9.5 * FPS)
    for name, o in objs.items():
        off = Vector(PARTS[name][1])
        base = o.location.copy()
        o.location = base; o.keyframe_insert("location", frame=t0)
        # the parent turns the device 180 deg about x: offsets are given in world axes, so convert
        o.location = base + Vector((off.x, -off.y, -off.z)); o.keyframe_insert("location", frame=t1)
    for fc in [fc for o in objs.values() if o.animation_data for fc in o.animation_data.action.fcurves] if hasattr(bpy.types.Action, "fcurves") else []:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
    objs["Leds_L"] and None
    for n in ("Leds_L", "Leds_R", "RingLeds"):
        if n in objs:
            pass
    emit_socket(M["led"]).default_value = 3.0
    emit_socket(M["ledR"]).default_value = 3.0
    emit_socket(M["ring"]).default_value = 3.0
elif SHOT in ("burst", "uv"):
    for n in ("InspectionHood", "RingLeds"):
        if n in objs:
            objs[n].hide_render = True
    tgt.location = (0, -0.01, -0.13); cam.location = (0, -0.62, 0.12); camd.lens = 32
    key_rot(rig, 1, -28); key_rot(rig, F, -10)
    key.data.energy = 4; rim.data.energy = 3; top.data.energy = 1
    # one spot per pod, aimed at the plate centre (toe-in like the pods)
    def spot(name, x):
        l = bpy.data.lights.new(name, "SPOT"); l.spot_size = math.radians(80); l.spot_blend = 0.6; l.energy = 0
        o = bpy.data.objects.new(name, l); o.location = (x, -0.014, -0.05); sc.collection.objects.link(o)
        c = o.constraints.new("TRACK_TO"); c.target = tgt2; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
        return l
    tgt2 = bpy.data.objects.new("tgt2", None); tgt2.location = (0, -0.014, PLATE_Z); sc.collection.objects.link(tgt2)
    sl, sr = spot("spotL", 0.045), spot("spotR", -0.045)
    led = emit_socket(M["led"]); ledr = emit_socket(M["ledR"])
    if SHOT == "burst":
        seq = [("B1", 1, 1), ("L", 1, 0), ("A", 0, 0), ("R", 0, 1), ("B2", 1, 1)]
        step = int(0.9 * FPS)
        f = 1
        for cycle in range(3):
            for name, a, b in seq:
                for fr in (f, f + step - 2):
                    sl.energy = 6.0 * a; sr.energy = 6.0 * b
                    sl.keyframe_insert("energy", frame=fr); sr.keyframe_insert("energy", frame=fr)
                    led.default_value = 12.0 * a; led.keyframe_insert("default_value", frame=fr)
                    ledr.default_value = 12.0 * b; ledr.keyframe_insert("default_value", frame=fr)
                for fr in (f + step - 1,):
                    sl.energy = 0; sr.energy = 0; sl.keyframe_insert("energy", frame=fr); sr.keyframe_insert("energy", frame=fr)
                    led.default_value = 0; led.keyframe_insert("default_value", frame=fr)
                    ledr.default_value = 0; ledr.keyframe_insert("default_value", frame=fr)
                f += step
        json.dump({"step_frames": step, "seq": [s[0] for s in seq]}, open(OUT + ".json", "w"))
    else:
        uvc = (0.35, 0.12, 1.0)
        emit_color(M["led"]).default_value = (*uvc, 1); emit_color(M["ledR"]).default_value = (*uvc, 1)
        for l in (sl, sr):
            l.color = uvc
        for fr, e in ((1, 0), (int(0.8 * FPS), 0), (int(1.4 * FPS), 1), (F, 1)):
            sl.energy = sr.energy = 4.0 * e; sl.keyframe_insert("energy", frame=fr); sr.keyframe_insert("energy", frame=fr)
            led.default_value = 14.0 * e; led.keyframe_insert("default_value", frame=fr)
            ledr.default_value = 14.0 * e; ledr.keyframe_insert("default_value", frame=fr)
        # the plate under UV: dark, the penetrant in the crack glows (illustration texture)
        pm = bpy.data.materials["plate"]; b = pm.node_tree.nodes["Principled BSDF"]
        t = pm.node_tree.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(os.path.join(V, "plate_uv_glow.png"))
        pm.node_tree.links.new(t.outputs["Color"], b.inputs["Emission Color"])
        es = b.inputs["Emission Strength"]
        for fr, e in ((1, 0), (int(1.4 * FPS), 0), (int(2.4 * FPS), 6), (F, 6)):
            es.default_value = e; es.keyframe_insert("default_value", frame=fr)
elif SHOT == "surface":
    data = json.load(open(os.path.join(A, "surface_dent.json")))
    it = next(i for i in data["items"] if i["type"] == "dent")
    H = it["height_mm"]; s = it["spacing_mm"]; ny, nx = len(H), len(H[0])
    EX = 6.0
    import bmesh
    me = bpy.data.meshes.new("surf"); bm = bmesh.new()
    vs = [[bm.verts.new(((j - (nx - 1) / 2) * s / 1000, (i - (ny - 1) / 2) * s / 1000, H[i][j] * EX / 1000)) for j in range(nx)] for i in range(ny)]
    for i in range(ny - 1):
        for j in range(nx - 1):
            bm.faces.new((vs[i][j], vs[i][j + 1], vs[i + 1][j + 1], vs[i + 1][j]))
    bm.to_mesh(me); bm.free()
    so = bpy.data.objects.new("surf", me); sc.collection.objects.link(so)
    for p in me.polygons:
        p.use_smooth = True
    hmin = min(min(r) for r in H); hmax = max(max(r) for r in H)
    col = me.color_attributes.new("h", "FLOAT_COLOR", "POINT")
    for k, v in enumerate(me.vertices):
        t = (v.co.z * 1000 / EX - hmin) / max(hmax - hmin, 1e-6)
        t0 = max(0.0, min(1.0, (v.co.z * 1000 / EX) / hmin)) if hmin < 0 else 0.0     # 0 at the surface, 1 at the deepest point
        c_top, c_deep = (0.42, 0.46, 0.52), (0.02, 0.18, 0.75)
        c = tuple(c_top[q] + (c_deep[q] - c_top[q]) * (t0 ** 0.7) for q in range(3))
        col.data[k].color = (*c, 1)
    sm = bpy.data.materials.new("surf"); sm.use_nodes = True
    an = sm.node_tree.nodes.new("ShaderNodeVertexColor"); an.layer_name = "h"
    sm.node_tree.links.new(an.outputs["Color"], sm.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    sm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.35
    sm.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.3
    me.materials.append(sm)
    size = max(nx, ny) * s / 1000
    tgt.location = (0, 0, -0.0005); cam.location = (0, -1.7 * size, 1.0 * size); camd.lens = 40; camd.clip_start = 0.0005
    key.location = (-1.5 * size, -1.0 * size, 2.0 * size); key.data.size = size; key.data.energy = 0.25
    rim.location = (1.5 * size, 1.5 * size, 1.0 * size); rim.data.size = size; rim.data.energy = 0.12; top.data.energy = 0.06
    key_rot(rig, 1, -60); key_rot(rig, F, 60)
    json.dump({"depth_mm": it["depth_mm"], "spacing_mm": s, "size_mm": size * 1000, "exaggeration": EX}, open(OUT + ".json", "w"))

# label anchors for the 2D overlay (screen px of each labelled part's centre, every frame)
if SHOT == "explode":
    from bpy_extras.object_utils import world_to_camera_view
    rec = {}
    for fr in range(1, F + 1, 3):
        sc.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        row = {}
        for n in LABELS:
            o = objs.get(n)
            if o is None:
                continue
            oe = o.evaluated_get(dg)
            c = sum((oe.matrix_world @ Vector(b) for b in oe.bound_box), Vector()) / 8
            p = world_to_camera_view(sc, cam, c)
            row[n] = [round(p.x * 1920, 1), round((1 - p.y) * 1080, 1)]
        rec[fr] = row
    json.dump({"labels": LABELS, "anchors": rec}, open(OUT + ".json", "w"))

if STILL:
    sc.frame_set(STILL); sc.render.filepath = OUT; sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
else:
    import os
    fdir = OUT + "_frames"; os.makedirs(fdir, exist_ok=True)
    sc.render.filepath = fdir + "/f_"
    sc.render.image_settings.file_format = "JPEG"; sc.render.image_settings.quality = 93
    bpy.ops.render.render(animation=True)
print("DONE", SHOT, OUT)
