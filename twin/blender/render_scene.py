"""Build and render Ayas twin scenes with Blender/Cycles. Runs INSIDE Blender.

    Blender -b --factory-startup -P twin/blender/render_scene.py -- \
        --scenes WORK [WORK ...] [--device METAL|CPU|OPTIX|CUDA] [--samples N] [--res-scale F] \
        [--overview] [--overview-samples N]

Each WORK dir holds plan.json (sampler), the texture maps (textures.py) and
blender_cam.json (ayas.twin.render.prepare). Per scene this writes render.exr
(multilayer, half float, linear, light groups L/R/UV/AMB), render_clean.exr when
plan.render.clean_pass, render_meta.json and blender.log. One bad scene never stops
the batch: its error goes into its render_meta.json. --overview also writes overview.png:
the same scene from a third-person camera (device shell, glowing pods, lamps), for demos.

Lighting (docs/twin/DESIGN_STEEL.md C, D): ambient.lights (area RECTANGLE/DISK, point,
spot, sun; W and W/m^2 at 683 lm/W), the world and ambient.occluders (grey boxes) are all
in light group AMB; the v1 ambient.ceiling list is still read. extras.hood is a matte-black
hollow frustum built in the camera frame (see build_hood), visible and casting shadows.

Split render: with one render, Cycles' light tree gives most samples to a bright sun or
window, and the L/R passes get noisier the brighter the ambient (measured: 28 % rms per
pixel in C6 against 7 % in the dark at 64 spp). That is render noise, not physics, so when
the scene has ambient light it is rendered twice: render.exr with the LEDs only (world
strength 0, ambient lights hidden) and render_amb.exr with the ambient only (LEDs hidden).
ayas.twin.render merges the AMB pass (and its light in Combined) into render.exr and
deletes render_amb.exr. plan.render.split_ambient = false renders once instead;
plan.render.amb_samples sets the samples of the ambient pass (default: the same).

Frames and units follow docs/twin/DESIGN.md section 1: world metres, X = u/1000,
Y = -v/1000, Z = w/1000; the plate front face is Z = 0. Plate-frame angles (holes,
grain, gloves, rivet tilt direction) are measured from +u towards +v, so the world
direction of angle a is (cos a, -sin a, 0).

Only bpy, bmesh, mathutils and the stdlib are used: never import ayas here.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

GROUPS = ("L", "R", "UV", "AMB")
# Geometry.Normal.z above this is the plate front face. DESIGN says 0.9, but the first
# face of a 2-segment 90-degree bevel has z = cos(22.5 deg) = 0.924, so 0.95 keeps the
# maps off the bevel and still takes every truly flat front face (z = 1).
FRONT_Z = 0.95
SINK_M = 5e-6            # heads sit 5 um into their surface: no hairline gap, no coplanar faces
BG_GAP_M = 5e-4          # background plane this far below the plate back face
BG_SIZE_M = 2.0
DEFAULT_TEX = {"normal": "normal.png", "albedo": "albedo.png", "roughness": "rough.png",
               "metallic": "metal.png", "uv_emit": None}
NON_COLOR = {"normal", "roughness", "metallic", "uv_emit"}
UV_EMIT_FILE = "uvemit.png"   # DESIGN 4: written by textures only when something fluoresces
STEEL = {"base_color": [0.42, 0.43, 0.45], "roughness": 0.3}
MANDREL = {"base_color": [0.55, 0.56, 0.58], "roughness": 0.3}
NITRILE = {"base_color": [0.012, 0.012, 0.014], "roughness": 0.45}
FLUOR_COLOR = (0.55, 1.0, 0.35)
GRAIN_TO_TANGENT_DEG = 90.0   # see tangent(): highlight streaks across the grain, as on real brushed sheet
HOOD_TOP_Z_M = 0.017          # hood top: camera-frame z of the pod fronts (extras.hood.top_z_cam_mm)
HOOD_LID_Z_M = -0.001         # the hood is closed by a lid just behind the lens (light-tight round the pods)
OVERVIEW_RES = (1280, 720)
OVERVIEW_DIST_M = 0.75
OVERVIEW_ELEV_DEG = 12.0
LM_PER_W = 683.0


# ---------------------------------------------------------------- small helpers

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="render_scene.py")
    p.add_argument("--scenes", nargs="+", required=True, help="scene work directories")
    p.add_argument("--device", default="METAL", help="METAL | CPU | OPTIX | CUDA | HIP | ONEAPI")
    p.add_argument("--samples", type=int, default=None, help="override plan.render.samples")
    p.add_argument("--res-scale", type=float, default=1.0, help="scale the ROI resolution, same field of view")
    p.add_argument("--overview", action="store_true", help="also render overview.png from a third-person camera")
    p.add_argument("--overview-samples", type=int, default=32, help="Cycles samples of the (denoised) overview")
    return p.parse_args(argv)


def scaled_size(w: int, h: int, f: float) -> tuple[int, int]:
    """Same rounding as ayas.twin.render.scaled_size."""
    return max(1, int(math.floor(w * f + 0.5))), max(1, int(math.floor(h * f + 0.5)))


def render_K(lens_mm: float, sensor_mm: float, shift_x: float, shift_y: float, w: int, h: int) -> list:
    """OpenCV K (pixel centres on integers) of the image Blender renders (sensor_fit AUTO)."""
    view = max(w, h)
    f = lens_mm / sensor_mm * view
    return [[f, 0.0, w / 2 - 0.5 - shift_x * view], [0.0, f, h / 2 - 0.5 + shift_y * view], [0.0, 0.0, 1.0]]


def world_dir(angle_deg: float) -> tuple[float, float]:
    """World XY direction of a plate-frame angle (from +u towards +v)."""
    a = math.radians(angle_deg)
    return math.cos(a), -math.sin(a)


def rgba(c, default=(0.8, 0.8, 0.8)) -> tuple:
    c = list(c if c is not None else default)
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)


def link(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_object(name: str, bm: bmesh.types.BMesh, smooth_angle: float | None = None) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth_angle is not None:
        me.shade_smooth()
        me.set_sharp_from_angle(angle=smooth_angle)
    return link(bpy.data.objects.new(name, me))


def add_box(bm, x0, x1, y0, y1, z0, z1) -> None:
    vs = [bm.verts.new((x, y, z)) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    v = lambda i, j, k: vs[i * 4 + j * 2 + k]  # noqa: E731
    for quad in (((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
                 ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
                 ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))):
        bm.faces.new([v(*q) for q in quad])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])


def add_elliptic_prism(bm, cx, cy, a, b, ang_rad, z0, z1, n) -> None:
    """Closed elliptic cylinder (semi-axes a along world angle ang_rad, b across), smooth walls."""
    ca, sa = math.cos(ang_rad), math.sin(ang_rad)
    bot, top = [], []
    for k in range(n):
        th = 2 * math.pi * k / n
        x, y = a * math.cos(th), b * math.sin(th)
        X, Y = cx + x * ca - y * sa, cy + x * sa + y * ca
        bot.append(bm.verts.new((X, Y, z0)))
        top.append(bm.verts.new((X, Y, z1)))
    walls = [bm.faces.new((bot[k], bot[(k + 1) % n], top[(k + 1) % n], top[k])) for k in range(n)]
    for f in walls:
        f.smooth = True
    caps = [bm.faces.new(top), bm.faces.new(bot[::-1])]
    bmesh.ops.recalc_face_normals(bm, faces=walls + caps)


def revolve(bm, profile, n, mat_index=None) -> None:
    """Closed surface of revolution about Z from (r, z) profile points (r = 0 makes a pole).

    mat_index(i) gives the material index of the band between profile points i and i+1.
    """
    rings = []
    for r, z in profile:
        if r <= 1e-12:
            rings.append([bm.verts.new((0.0, 0.0, z))])
        else:
            rings.append([bm.verts.new((r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z))
                          for k in range(n)])
    faces = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for k in range(n):
            k1 = (k + 1) % n
            if len(a) == 1 and len(b) == 1:
                break
            if len(a) == 1:
                vs = (a[0], b[k], b[k1])
            elif len(b) == 1:
                vs = (a[k], a[k1], b[0])
            else:
                vs = (a[k], a[k1], b[k1], b[k])
            f = bm.faces.new(vs)
            f.material_index = mat_index(i) if mat_index else 0
            faces.append(f)
    bmesh.ops.recalc_face_normals(bm, faces=faces)


def apply_modifiers(obj: bpy.types.Object) -> None:
    """Bake the modifier stack into the mesh (no operator context needed).

    The baked mesh comes back with stray (empty) material slots from the boolean, so the
    slots are cleared: callers append materials afterwards (face material indices are kept).
    """
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    me.materials.clear()
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    bpy.data.meshes.remove(old)


def load_image(path: Path, colorspace: str) -> bpy.types.Image:
    if not path.is_file():
        raise FileNotFoundError(f"texture map missing: {path}")
    img = bpy.data.images.load(str(path), check_existing=False)
    img.colorspace_settings.name = colorspace
    return img


@contextmanager
def log_to(path: Path):
    """Send this process's stdout/stderr (C level too, so Cycles output) to path."""
    libc = None
    try:
        libc = ctypes.CDLL(None)
    except OSError:
        pass

    def flush():
        sys.stdout.flush()
        sys.stderr.flush()
        if libc is not None:
            libc.fflush(None)

    flush()
    f = open(path, "w")
    saved = os.dup(1), os.dup(2)
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    try:
        yield
    finally:
        flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        f.close()


# ---------------------------------------------------------------- materials

def new_material(name: str):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    return mat, nt, nt.nodes.new("ShaderNodeOutputMaterial")


def principled(nt, base, metallic: float, rough: float, aniso: float = 0.0):
    """Principled BSDF. Anisotropic Rotation stays 0: the direction comes only from tangent()."""
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = rgba(base)
    b.inputs["Metallic"].default_value = float(metallic)
    b.inputs["Roughness"].default_value = float(rough)
    b.inputs["Anisotropic"].default_value = float(aniso)
    b.inputs["Anisotropic Rotation"].default_value = 0.0
    return b


def grain_of(spec: dict, default: float = 0.0) -> float:
    """Plate-frame grain (brush line) direction of a material spec, in degrees.

    grain_dir_deg when given (textures draw the grain along it). Otherwise it is derived from
    aniso_rotation as the sampler does (Blender turns +X counter-clockwise about +Z, 1.0 =
    360 deg, i.e. from +u towards -v). aniso_rotation is never applied on top of
    grain_dir_deg: the sampler writes both for the same direction.
    """
    if spec.get("grain_dir_deg") is not None:
        return float(spec["grain_dir_deg"])
    if spec.get("aniso_rotation") is not None:
        return (-360.0 * float(spec["aniso_rotation"])) % 180.0
    return default


def tangent(nt, grain_deg: float):
    """Constant world-space anisotropy tangent for grain (polishing lines) along grain_deg.

    Cycles widens the roughness ALONG the tangent, so a highlight stretches along it
    (checked by rendering). Brushed metal streaks run ACROSS the grooves, so the tangent is
    the grain direction turned by GRAIN_TO_TANGENT_DEG.
    """
    c = nt.nodes.new("ShaderNodeCombineXYZ")
    c.inputs["X"].default_value, c.inputs["Y"].default_value = world_dir(grain_deg + GRAIN_TO_TANGENT_DEG)
    c.inputs["Z"].default_value = 0.0
    return c.outputs["Vector"]


def simple_material(name: str, spec: dict, metallic: float = 1.0, grain_deg: float = 0.0):
    """Plain Principled material; its tangent follows the spec's own grain, else grain_deg."""
    mat, nt, out = new_material(name)
    b = principled(nt, spec.get("base_color"), spec.get("metallic", metallic), spec.get("roughness", 0.3),
                   spec.get("anisotropy", 0.0))
    nt.links.new(tangent(nt, grain_of(spec, grain_deg)), b.inputs["Tangent"])
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    return mat


def developer_material(dev: dict):
    a = float(dev.get("albedo", 0.88))
    return simple_material("developer", {"base_color": [a, a, a], "roughness": dev.get("roughness", 0.9)},
                           metallic=0.0)


def diffuse_material(name: str, color, rough: float = 1.0):
    mat, nt, out = new_material(name)
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = rgba(color)
    d.inputs["Roughness"].default_value = rough
    nt.links.new(d.outputs["BSDF"], out.inputs["Surface"])
    return mat


def plate_material(plan: dict, work: Path, plate_obj, developer: dict | None):
    """Texture maps on the front face (Normal.z > FRONT_Z), plain metal (or developer) elsewhere.

    Returns (material, {map key: image node}) so a clean pass can swap the images.
    """
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    pm = plan["plate"].get("material", {})
    grain = grain_of(pm)
    tex = dict(DEFAULT_TEX, **(plan.get("textures") or {}))
    if not tex.get("uv_emit") and (work / UV_EMIT_FILE).is_file():
        tex["uv_emit"] = UV_EMIT_FILE                      # textures writes it only when something fluoresces
    mat, nt, out = new_material("plate")
    N, L = nt.nodes, nt.links

    tc = N.new("ShaderNodeTexCoord")
    tc.object = plate_obj                                  # identity transform: Object = world
    mul = N.new("ShaderNodeVectorMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = (1000.0 / W, 1000.0 / H, 0.0)
    add = N.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.0, 1.0, 0.0)         # U = X*1000/W, V = 1 + Y*1000/H
    L.new(tc.outputs["Object"], mul.inputs[0])
    L.new(mul.outputs["Vector"], add.inputs[0])

    nodes = {}

    def image(key):
        n = N.new("ShaderNodeTexImage")
        n.image = load_image(work / tex[key], "Non-Color" if key in NON_COLOR else "sRGB")
        n.interpolation = "Linear"
        n.extension = "EXTEND"
        L.new(add.outputs["Vector"], n.inputs["Vector"])
        nodes[key] = n
        return n.outputs["Color"]

    # developer hides the brushed grain, so a penetrant front face is not anisotropic
    front = principled(nt, pm.get("base_color"), 1.0, pm.get("roughness", 0.3),
                       0.0 if developer is not None else pm.get("anisotropy", 0.0))
    L.new(image("albedo"), front.inputs["Base Color"])
    L.new(image("metallic"), front.inputs["Metallic"])
    L.new(image("roughness"), front.inputs["Roughness"])
    nmap = N.new("ShaderNodeNormalMap")
    nmap.space = "OBJECT"
    L.new(image("normal"), nmap.inputs["Color"])
    L.new(nmap.outputs["Normal"], front.inputs["Normal"])
    L.new(tangent(nt, grain), front.inputs["Tangent"])
    front_out = front.outputs["BSDF"]

    if tex.get("uv_emit"):                                 # fluorescence: emission, plate lightgroup = UV
        em = N.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*FLUOR_COLOR, 1.0)
        k = N.new("ShaderNodeMath")
        k.operation = "MULTIPLY"
        k.inputs[1].default_value = float((plan.get("render") or {}).get("uv_emit_strength", 1.0))
        L.new(image("uv_emit"), k.inputs[0])
        L.new(k.outputs["Value"], em.inputs["Strength"])
        both = N.new("ShaderNodeAddShader")
        L.new(front_out, both.inputs[0])
        L.new(em.outputs["Emission"], both.inputs[1])
        front_out = both.outputs["Shader"]

    if developer is not None:
        a = float(developer.get("albedo", 0.88))
        plain = principled(nt, [a, a, a], 0.0, developer.get("roughness", 0.9))
    else:
        plain = principled(nt, pm.get("base_color"), pm.get("metallic", 1.0), pm.get("roughness", 0.3))
    L.new(tangent(nt, grain), plain.inputs["Tangent"])

    geo = N.new("ShaderNodeNewGeometry")
    sep = N.new("ShaderNodeSeparateXYZ")
    gt = N.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = FRONT_Z
    L.new(geo.outputs["Normal"], sep.inputs[0])
    L.new(sep.outputs["Z"], gt.inputs[0])
    mix = N.new("ShaderNodeMixShader")
    L.new(gt.outputs["Value"], mix.inputs[0])
    L.new(plain.outputs["BSDF"], mix.inputs[1])
    L.new(front_out, mix.inputs[2])
    L.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat, nodes, tex


def swap_to_clean(nodes: dict, tex: dict, work: Path, meta: dict) -> None:
    for key, node in nodes.items():
        p = Path(tex[key])
        clean = work / f"{p.stem}_clean{p.suffix}"
        if clean.is_file():
            node.image = load_image(clean, "Non-Color" if key in NON_COLOR else "sRGB")
        else:
            meta["warnings"].append(f"clean pass: {clean.name} missing, kept {p.name}")


# ---------------------------------------------------------------- geometry

def cutter_object(holes: list, z0: float, z1: float):
    bm = bmesh.new()
    for h in holes:
        d = float(h["d_mm"]) / 1000.0
        a = 0.5 * d * float(h.get("ovality", 1.0) or 1.0)   # d_mm = minor diameter, major = d * ovality
        x, y = float(h["u"]) / 1000.0, -float(h["v"]) / 1000.0
        n = 64 if d <= 0.008 else 128
        add_elliptic_prism(bm, x, y, a, 0.5 * d, -math.radians(float(h.get("angle_deg", 0.0))), z0, z1, n)
    return mesh_object("cutter", bm)


def cut_and_bevel(obj, cutter, bevel_m: float) -> None:
    if bevel_m > 0:
        bv = obj.modifiers.new("bevel", "BEVEL")
        bv.width = bevel_m
        bv.segments = 2
        bv.limit_method = "ANGLE"
        bv.angle_limit = math.radians(30.0)
    if cutter is not None:
        bo = obj.modifiers.new("holes", "BOOLEAN")
        bo.operation = "DIFFERENCE"
        bo.solver = "EXACT"
        bo.object = cutter
    apply_modifiers(obj)
    if len(obj.data.polygons) == 0:
        raise RuntimeError(f"{obj.name}: boolean produced an empty mesh")


def build_plate(plan: dict, cutter):
    W, H = (float(x) / 1000.0 for x in plan["plate"]["size_mm"])
    t = float(plan["plate"]["thickness_mm"]) / 1000.0
    bm = bmesh.new()
    add_box(bm, 0.0, W, -H, 0.0, -t, 0.0)
    obj = mesh_object("plate", bm)
    cut_and_bevel(obj, cutter, float(plan["plate"].get("edge_bevel_mm", 0.0)) / 1000.0)
    return obj


def build_sheets(plan: dict, cutter, mat_for) -> list:
    out = []
    bevel = float(plan["plate"].get("edge_bevel_mm", 0.0)) / 1000.0
    for s in plan.get("sheets") or []:
        u0, v0, w, h = (float(x) for x in s["rect_mm"])
        ts = float(s["thickness_mm"]) / 1000.0
        bm = bmesh.new()
        add_box(bm, u0 / 1000.0, (u0 + w) / 1000.0, -(v0 + h) / 1000.0, -v0 / 1000.0, 0.0, ts)
        obj = mesh_object(f"sheet_{s['id']}", bm)
        cut_and_bevel(obj, cutter, min(bevel, 0.4 * ts))
        obj.data.materials.append(mat_for(s))
        out.append(obj)
    return out


def surface_z(plan: dict, u: float, v: float, sheet_id=None) -> float:
    """Top surface height (m) at plate point (u, v): the thickest sheet covering it, else 0."""
    z = 0.0
    for s in plan.get("sheets") or []:
        u0, v0, w, h = (float(x) for x in s["rect_mm"])
        if (sheet_id is not None and s.get("id") == sheet_id) or (u0 <= u <= u0 + w and v0 <= v <= v0 + h):
            z = max(z, float(s["thickness_mm"]) / 1000.0)
    return z


def dome_profile(D: float, h: float, dm: float) -> list:
    """(r, z) in metres: blind-rivet dome head with a mandrel recess (dm wide, 0.6 h deep)."""
    R, e = D / 2, 0.12 * h
    rho = (R * R + (h - e) ** 2) / (2 * (h - e))          # sphere through (R, e) and (0, h)
    zc = lambda r: h - rho + math.sqrt(max(rho * rho - r * r, 0.0))  # noqa: E731
    rm = min(dm / 2, 0.45 * R)
    prof = [(0.0, 0.0), (R, 0.0), (R, e)]
    prof += [(R - (R - rm) * i / 16, zc(R - (R - rm) * i / 16)) for i in range(1, 17)]
    bottom = max(zc(rm) - 0.6 * h, 0.05 * h)
    prof += [(rm, bottom), (0.0, bottom)]
    return prof


def pan_profile(D: float, h: float) -> list:
    """(r, z) in metres: pan head, vertical side to 0.55 h then a quarter-ellipse to a flat top."""
    R = D / 2
    prof = [(0.0, 0.0), (R, 0.0), (R, 0.55 * h)]
    for i in range(1, 11):
        a = 0.5 * math.pi * i / 10
        prof.append((0.8 * R + 0.2 * R * math.cos(a), 0.55 * h + 0.45 * h * math.sin(a)))
    prof.append((0.0, h))
    return prof


def build_rivet(rv: dict, hole: dict, plan: dict, mats: dict):
    """Head (revolved profile) on its surface, tilted by tilt_deg towards tilt_dir_deg, plus a shank."""
    D = float(rv["head_d_mm"]) / 1000.0
    h = float(rv["head_h_mm"]) / 1000.0
    screw = rv.get("kind") == "screw"
    u, v = float(hole["u"]), float(hole["v"])
    X, Y = u / 1000.0, -v / 1000.0
    z_top = surface_z(plan, u, v, hole.get("sheet"))
    t = float(plan["plate"]["thickness_mm"]) / 1000.0

    bm = bmesh.new()
    if screw:
        revolve(bm, pan_profile(D, h), 96)
    else:
        prof = dome_profile(D, h, float(rv.get("mandrel_d_mm", 0.27 * rv["head_d_mm"])) / 1000.0)
        n_cap = len(prof) - 2
        revolve(bm, prof, 96, mat_index=lambda i: 1 if i >= n_cap else 0)   # recess floor: mandrel steel
    head = mesh_object(f"rivet_{rv['id']}", bm, smooth_angle=math.radians(35.0))
    if screw:
        bm = bmesh.new()
        sw = 0.18 * D
        add_box(bm, -0.7 * D, 0.7 * D, -sw / 2, sw / 2, 0.55 * h, 2 * h)
        slot = mesh_object("slot", bm)
        slot.rotation_euler = (0.0, 0.0, -math.radians(float(rv.get("slot_deg", 0.0))))
        bo = head.modifiers.new("slot", "BOOLEAN")
        bo.operation, bo.solver, bo.object = "DIFFERENCE", "EXACT", slot
        apply_modifiers(head)
        bpy.data.objects.remove(slot)
    head.data.materials.append(mats["screw" if screw else "rivet"])
    head.data.materials.append(mats["mandrel"])                      # material index 1

    tilt = math.radians(float(rv.get("tilt_deg", 0.0)))
    dx, dy = world_dir(float(rv.get("tilt_dir_deg", 0.0)))
    head.rotation_mode = "QUATERNION"
    head.rotation_quaternion = Quaternion((-dy, dx, 0.0), tilt)  # axis z x d: the head leans down towards d
    head.location = (X, Y, z_top - SINK_M + 0.5 * D * math.sin(abs(tilt)))

    bm = bmesh.new()
    rs = 0.5 * 0.98 * float(hole["d_mm"]) / 1000.0
    add_elliptic_prism(bm, X, Y, rs, rs, 0.0, -t, z_top + 0.1 * h, 48)   # top hidden in the head
    shank = mesh_object(f"shank_{rv['id']}", bm)
    shank.data.materials.append(mats["screw" if screw else "rivet"])
    return head, shank


def build_background(plan: dict):
    bg = (plan.get("extras") or {}).get("background") or {}
    W, H = (float(x) / 1000.0 for x in plan["plate"]["size_mm"])
    z = -float(plan["plate"]["thickness_mm"]) / 1000.0 - BG_GAP_M
    cx, cy, s = W / 2, -H / 2, BG_SIZE_M / 2
    bm = bmesh.new()
    vs = [bm.verts.new((cx + a * s, cy + b * s, z)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    bm.faces.new(vs)                                       # counter-clockwise from above: normal +Z
    obj = mesh_object("background", bm)

    mat, nt, out = new_material("background")
    color = bg.get("color", [0.2, 0.2, 0.22])
    b = principled(nt, color, 0.0, bg.get("roughness", 0.8))
    if bg.get("kind", "noise") == "noise":
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = float(bg.get("scale", 40.0))   # per metre of world
        nz.inputs["Detail"].default_value = 6.0
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        a_in, b_in = (sk for sk in mix.inputs if sk.type == "RGBA")   # A and B exist once per data type
        a_in.default_value = rgba([0.5 * c for c in color])
        b_in.default_value = rgba([min(1.0, 1.5 * c) for c in color])
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        nt.links.new(nz.outputs["Factor"], mix.inputs[0])
        nt.links.new(next(sk for sk in mix.outputs if sk.type == "RGBA"), b.inputs["Base Color"])
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def glove_axis(g: dict) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """(centre (u, v) mm, unit direction (du, dv), length mm, radius mm) of a glove finger.

    (u, v) is the centre of the rounded fingertip; the finger runs from there along
    angle_deg, length_mm long overall (tip included). The sampler puts (u, v) r - (0..4) mm
    outside a plate edge, so the finger overlaps the plate by up to 4 mm.
    """
    L, r = float(g["length_mm"]), float(g["radius_mm"])
    a = math.radians(float(g.get("angle_deg", 0.0)))
    du, dv = math.cos(a), math.sin(a)
    k = max(L / 2 - r, 0.0)
    return (float(g["u"]) + du * k, float(g["v"]) + dv * k), (du, dv), L, r


def glove_support(plan: dict, centre, d, L: float, r: float) -> float:
    """Height (m) the finger rests on: the plate or a sheet if its footprint overlaps them, else the background."""
    W, H = (float(x) for x in plan["plate"]["size_mm"])
    support = -float(plan["plate"]["thickness_mm"]) / 1000.0 - BG_GAP_M
    half = max(L / 2 - r, 0.0)
    for i in range(-8, 9):
        for j in (-1.0, -0.5, 0.0, 0.5, 1.0):
            s, q = half * i / 8, r * j
            u, v = centre[0] + d[0] * s - d[1] * q, centre[1] + d[1] * s + d[0] * q
            if 0.0 <= u <= W and 0.0 <= v <= H:
                support = max(support, surface_z(plan, u, v))
    for e in (-1.0, 1.0):                                  # the two rounded ends
        u, v = centre[0] + d[0] * e * (half + r), centre[1] + d[1] * e * (half + r)
        if 0.0 <= u <= W and 0.0 <= v <= H:
            support = max(support, surface_z(plan, u, v))
    return support


def build_gloves(plan: dict) -> list:
    """Nitrile fingers (black unless the glove has a color): capsules resting on what is below."""
    gloves = (plan.get("extras") or {}).get("gloves") or []
    if not gloves:
        return []
    out = []
    for i, g in enumerate(gloves):
        (uc, vc), d, L, r = glove_axis(g)
        a = math.atan2(d[1], d[0])
        support = glove_support(plan, (uc, vc), d, L, r)
        mat = simple_material(f"nitrile_{i}", dict(NITRILE, base_color=g.get("color") or NITRILE["base_color"]),
                              metallic=0.0)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r / 1000.0)
        half = max(L / 2 - r, 0.0) / 1000.0
        for vert in bm.verts:
            if vert.co.x > 1e-9:
                vert.co.x += half
            elif vert.co.x < -1e-9:
                vert.co.x -= half
        obj = mesh_object(f"glove_{i}", bm, smooth_angle=math.radians(60.0))
        obj.location = (uc / 1000.0, -vc / 1000.0, support + r / 1000.0)
        obj.rotation_euler = (0.0, 0.0, -a)
        obj.data.materials.append(mat)
        out.append(obj)
    return out


def build_device_box(cam: dict):
    """Black box in the OpenCV camera frame: casts shadows, seen in reflections, never by the camera."""
    db = cam.get("device_box_m")
    if not db:
        return None
    bm = bmesh.new()
    add_box(bm, *db["x"], *db["y"], *db["z"])
    obj = mesh_object("device_box", bm)
    obj.matrix_world = Matrix(cam["cv2world"])
    obj.visible_camera = False
    obj.visible_transmission = False
    obj.data.materials.append(diffuse_material("device_black", [0.03, 0.03, 0.03]))
    return obj


# ---------------------------------------------------------------- lights, world, camera, render

def light_quaternion(L: dict) -> Quaternion:
    """Orientation with local -Z along L['dir']; local +X along L['x_dir'] when given (area lights)."""
    z = -Vector(L.get("dir", (0, 0, -1))).normalized()
    if L.get("x_dir") is not None:
        x = Vector(L["x_dir"])
        x = x - x.dot(z) * z
        if x.length > 1e-6:
            x.normalize()
            return Matrix((x, z.cross(x), z)).transposed().to_quaternion()
    return (-z).to_track_quat("-Z", "Y")


def make_light(L: dict, group: str) -> bpy.types.Object:
    """One plan light: kind point | spot (power_w, radius_m), area (power_w, size_m, shape), sun (strength, angle_deg)."""
    kind = str(L.get("kind", "spot")).upper()
    if kind not in ("POINT", "SPOT", "AREA", "SUN"):
        raise ValueError(f"light {L.get('name')}: unknown kind {L.get('kind')!r}")
    ld = bpy.data.lights.new(L["name"], kind)
    ld.color = tuple(float(c) for c in L.get("color", (1, 1, 1))[:3])
    if kind == "SUN":
        ld.energy = float(L.get("strength", L.get("power_w", 1.0)))     # W/m^2 facing the sun
        ld.angle = math.radians(float(L.get("angle_deg", 0.53)))
    else:
        ld.energy = float(L["power_w"])
    if kind in ("POINT", "SPOT"):
        ld.shadow_soft_size = float(L.get("radius_m", 0.001))
    if kind == "SPOT":
        ld.spot_size = math.radians(min(float(L.get("spot_deg", 120.0)), 180.0))
        ld.spot_blend = float(L.get("blend", 0.5))
    if kind == "AREA":
        shape = str(L.get("shape", "RECTANGLE")).upper()
        ld.shape = shape
        size = [float(x) for x in (list(L.get("size_m") or [0.1]) * 2)[:2]]
        ld.size = size[0]
        if shape in ("RECTANGLE", "ELLIPSE"):
            ld.size_y = size[1]
    obj = link(bpy.data.objects.new(L["name"], ld))
    obj.location = tuple(float(x) for x in L.get("pos_m", (0, 0, 10)))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = light_quaternion(L)
    obj.lightgroup = L.get("group") or group
    return obj


def build_lights(plan: dict) -> None:
    for L in plan.get("lights") or []:
        make_light(L, "AMB")


def build_occluders(amb: dict, meta: dict) -> list:
    """Diffuse grey boxes (a hand or arm): center_m, size_m, rot_deg (Blender XYZ Euler, degrees)."""
    out = []
    for i, o in enumerate(amb.get("occluders") or []):
        if str(o.get("kind", "box")) != "box":
            meta["warnings"].append(f"occluder {i}: kind {o.get('kind')!r} not supported, skipped")
            continue
        sx, sy, sz = (float(v) / 2 for v in o["size_m"])
        bm = bmesh.new()
        add_box(bm, -sx, sx, -sy, sy, -sz, sz)
        obj = mesh_object(f"occluder_{i}", bm)
        obj.location = tuple(float(v) for v in o["center_m"])
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = tuple(math.radians(float(a)) for a in o.get("rot_deg", (0, 0, 0)))
        a = float(o.get("albedo", 0.35))
        obj.data.materials.append(diffuse_material(f"occluder_{i}", [a, a, a]))
        out.append(obj)
    return out


def hood_points(hood: dict, cam: dict) -> dict:
    """World corners of the hood (same maths as ayas.twin.lighting.hood_corners).

    In the OpenCV camera frame the top rectangle top_mm sits at z = top_z_cam_mm and the
    bottom_mm rectangle at the depth where the optical axis meets the bottom plane
    Z = gap_mm / 1000; each edge from a top corner to its bottom corner is intersected with
    that plane. Outer rectangles are wall_mm larger on every side. The lid rectangles close
    the top just behind the lens. Corner order (-x,-y), (+x,-y), (+x,+y), (-x,+y).
    """
    M = Matrix(cam["cv2world"])
    C = M.translation
    zb = float(hood.get("gap_mm", 0.0)) / 1000.0
    zt = float(hood.get("top_z_cam_mm", HOOD_TOP_Z_M * 1000.0)) / 1000.0
    a = M.to_3x3() @ Vector((0.0, 0.0, 1.0))
    depth = (zb - C.z) / a.z
    wall = float(hood.get("wall_mm", 2.0)) / 1000.0
    out = {}
    for name, grow in (("inner", 0.0), ("outer", 2 * wall)):
        wt, ht = (float(x) / 1000.0 + grow for x in hood.get("top_mm", (170.0, 160.0)))
        wb, hb = (float(x) / 1000.0 + grow for x in hood.get("bottom_mm", (200.0, 190.0)))
        zl = HOOD_LID_Z_M - (wall if grow else 0.0)
        lid, top, bot = [], [], []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            T = M @ Vector((sx * wt / 2, sy * ht / 2, zt))
            B = M @ Vector((sx * wb / 2, sy * hb / 2, depth))
            lid.append(M @ Vector((sx * wt / 2, sy * ht / 2, zl)))
            top.append(T)
            bot.append(T + (zb - T.z) / (B.z - T.z) * (B - T))
        out[f"{name}_lid"], out[f"{name}_top"], out[f"{name}_bottom"] = lid, top, bot
    return out


def build_hood(plan: dict, cam: dict):
    """extras.hood: a closed matte-black cup (lid, collar round the pods, frustum to the bottom plane).

    It is an ordinary object: visible to the camera, casting shadows. Ambient light can only
    reach the plate through the gap under its rim.
    """
    ex = plan.get("extras") or {}
    hood = ex.get("hood")
    if not isinstance(hood, dict) or (ex.get("device") or {}).get("mode", "hood") == "open":
        return None
    P = hood_points(hood, cam)
    bm = bmesh.new()
    V = {k: [bm.verts.new(tuple(p)) for p in pts] for k, pts in P.items()}
    faces = [bm.faces.new(V["inner_lid"]), bm.faces.new(V["outer_lid"][::-1])]
    for side in ("inner", "outer"):
        for a, b in (("lid", "top"), ("top", "bottom")):
            A, B = V[f"{side}_{a}"], V[f"{side}_{b}"]
            faces += [bm.faces.new((A[k], A[(k + 1) % 4], B[(k + 1) % 4], B[k])) for k in range(4)]
    I, O = V["inner_bottom"], V["outer_bottom"]
    faces += [bm.faces.new((I[k], I[(k + 1) % 4], O[(k + 1) % 4], O[k])) for k in range(4)]
    bmesh.ops.recalc_face_normals(bm, faces=faces)
    obj = mesh_object("hood", bm)
    a = float(hood.get("albedo", 0.03))
    obj.data.materials.append(diffuse_material("hood_black", [a, a, a]))
    return obj


def build_world(plan: dict, meta: dict | None = None) -> None:
    amb = plan.get("ambient") or {}
    w = bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    if w.node_tree is None:
        w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = rgba(amb.get("world_color"), (0.0, 0.0, 0.0))
    bg.inputs["Strength"].default_value = float(amb.get("world_strength", 0.0))
    w.lightgroup = "AMB"
    for i, c in enumerate(amb.get("ceiling") or []):
        ld = bpy.data.lights.new(f"ceiling_{i}", "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = (float(x) for x in c["size_m"])
        ld.energy = float(c["power_w"])
        ld.color = tuple(float(x) for x in c.get("color", (1, 1, 1))[:3])
        obj = link(bpy.data.objects.new(f"ceiling_{i}", ld))
        obj.location = tuple(float(x) for x in c["pos_m"])
        obj.rotation_euler = (0.0, 0.0, math.radians(float(c.get("rot_deg", 0.0))))   # emits along -Z
        obj.lightgroup = "AMB"
    for L in amb.get("lights") or []:                      # DESIGN_STEEL D ambient lights (AMB)
        make_light(L, "AMB")
    build_occluders(amb, meta if meta is not None else {"warnings": []})


def build_camera(plan: dict, cam: dict, res_scale: float) -> list:
    pc = plan.get("camera") or {}
    cd = bpy.data.cameras.new("camera")
    cd.lens = float(cam["lens_mm"])
    cd.sensor_fit = cam.get("sensor_fit", "AUTO")
    cd.sensor_width = cd.sensor_height = float(cam["sensor_width_mm"])
    cd.shift_x, cd.shift_y = float(cam["shift_x"]), float(cam["shift_y"])
    cd.clip_start, cd.clip_end = 0.01, 10.0
    cd.dof.use_dof = True
    cd.dof.focus_distance = float(pc.get("focus_m", cam.get("focus_m", 0.2)))
    cd.dof.aperture_fstop = float(pc.get("fstop", cam.get("fstop", 1.79)))
    obj = link(bpy.data.objects.new("camera", cd))
    obj.matrix_world = Matrix(cam["matrix_world"])
    scn = bpy.context.scene
    scn.camera = obj
    w, h = scaled_size(int(cam["resolution"][0]), int(cam["resolution"][1]), res_scale)
    scn.render.resolution_x, scn.render.resolution_y = w, h
    scn.render.resolution_percentage = 100
    scn.render.pixel_aspect_x = scn.render.pixel_aspect_y = 1.0
    return render_K(cd.lens, cd.sensor_width, cd.shift_x, cd.shift_y, w, h)


def setup_device(want: str) -> str:
    """Use the GPUs of type `want`; fall back to CPU when there are none."""
    scn = bpy.context.scene
    want = (want or "CPU").upper()
    if want == "CPU":
        scn.cycles.device = "CPU"
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = want
        prefs.get_devices()
    except (TypeError, ValueError, RuntimeError) as e:
        print(f"device {want} unavailable ({e}); using CPU")
        scn.cycles.device = "CPU"
        return "CPU"
    gpus = [d for d in prefs.devices if d.type == want]
    for d in prefs.devices:
        d.use = d.type == want
    if not gpus:
        prefs.compute_device_type = "NONE"
        scn.cycles.device = "CPU"
        return "CPU"
    scn.cycles.device = "GPU"
    return f"{want}: " + ", ".join(d.name for d in gpus)


def setup_render(plan: dict, samples: int | None) -> int:
    scn = bpy.context.scene
    r = plan.get("render") or {}
    cy = scn.cycles
    n = int(samples or r.get("samples", 64))
    cy.samples = n
    thr = float(r.get("noise_threshold", 0.02))
    cy.use_adaptive_sampling = thr > 0
    if thr > 0:
        cy.adaptive_threshold = thr
    cy.max_bounces = int(r.get("max_bounces", 6))
    cy.glossy_bounces = int(r.get("glossy_bounces", 4))
    cy.diffuse_bounces = min(4, cy.max_bounces)
    cy.transmission_bounces = min(4, cy.max_bounces)
    cy.transparent_max_bounces = 2
    cy.volume_bounces = 0
    cy.caustics_reflective = cy.caustics_refractive = False
    cy.use_denoising = False                               # light-group passes are never denoised
    cy.filter_width = 1.5
    cy.seed = int(plan.get("seed", 0)) % 2147483647
    cy.use_animated_seed = False
    scn.render.film_transparent = False
    scn.render.use_compositing = False
    scn.render.use_sequencer = False
    scn.view_settings.view_transform = "Standard"          # EXR stays scene-linear anyway
    try:
        scn.view_settings.look = "None"
    except TypeError:
        pass
    scn.view_settings.exposure = 0.0
    scn.view_settings.gamma = 1.0
    s = scn.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "MULTI_LAYER_IMAGE"                 # must come before file_format (5.x)
    s.file_format = "OPEN_EXR_MULTILAYER"
    s.color_depth = "16"
    s.exr_codec = "ZIP"
    vl = scn.view_layers[0]
    vl.use_pass_combined = True
    lights = (plan.get("lights") or []) + ((plan.get("ambient") or {}).get("lights") or [])
    extra = sorted({L.get("group") for L in lights if L.get("group") and L["group"] not in GROUPS})
    for g in list(GROUPS) + extra:
        if g not in vl.lightgroups:
            vl.lightgroups.add(name=g)
    return n


def split_ambient(plan: dict) -> bool:
    """Render LEDs and ambient apart? Only when there is ambient light (see the module docstring)."""
    amb = plan.get("ambient") or {}
    world = float(amb.get("world_strength", 0.0) or 0.0) > 0 and any(float(c) > 0 for c in amb.get("world_color") or [0])
    lit = world or bool(amb.get("lights")) or bool(amb.get("ceiling"))
    return lit and bool((plan.get("render") or {}).get("split_ambient", True))


def render_lit(path: Path, plan: dict, meta: dict) -> None:
    """path with every light, or (split_ambient) path with the LEDs only plus <stem>_amb.exr with the ambient only."""
    if not split_ambient(plan):
        render_to(path, meta)
        return
    scn = bpy.context.scene
    lights = [o for o in scn.objects if o.type == "LIGHT"]
    amb = [o for o in lights if o.lightgroup == "AMB"]
    led = [o for o in lights if o.lightgroup != "AMB"]
    bg = next((n for n in scn.world.node_tree.nodes if n.type == "BACKGROUND"), None) if scn.world else None
    strength = bg.inputs["Strength"].default_value if bg else 0.0
    samples = scn.cycles.samples
    try:
        for o in amb:
            o.hide_render = True
        if bg:
            bg.inputs["Strength"].default_value = 0.0
        render_to(path, meta)
        for o in amb:
            o.hide_render = False
        if bg:
            bg.inputs["Strength"].default_value = strength
        for o in led:
            o.hide_render = True
        scn.cycles.samples = int((plan.get("render") or {}).get("amb_samples") or samples)
        render_to(path.with_name(f"{path.stem}_amb{path.suffix}"), meta)
    finally:
        for o in lights:
            o.hide_render = False
        if bg:
            bg.inputs["Strength"].default_value = strength
        scn.cycles.samples = samples
    meta["split_ambient"] = True


def render_to(path: Path, meta: dict) -> None:
    scn = bpy.context.scene
    if path.exists():
        path.unlink()
    scn.render.filepath = str(path)
    try:
        bpy.ops.render.render(write_still=True)
    except RuntimeError as e:
        if scn.cycles.device != "GPU":
            raise
        meta["warnings"].append(f"GPU render failed ({e}); retried on CPU")
        meta["device"] = "CPU (GPU failed)"
        scn.cycles.device = "CPU"
        bpy.ops.render.render(write_still=True)
    if not path.is_file():
        raise RuntimeError(f"Blender wrote no {path.name}")
    meta["files"].append(path.name)


def build_scene(plan: dict, cam: dict, work: Path, args, meta: dict):
    """Everything except the render settings; returns the plate image nodes for a clean pass."""
    pen = plan.get("penetrant") if plan.get("mode") == "penetrant" else None
    developer = (pen or {}).get("developer", {}) if pen is not None else None
    t = float(plan["plate"]["thickness_mm"]) / 1000.0
    ts_max = max([float(s["thickness_mm"]) / 1000.0 for s in plan.get("sheets") or []] + [0.0])
    holes = plan.get("holes") or []
    cutter = cutter_object(holes, -t - 0.001, ts_max + 0.001) if holes else None

    plate = build_plate(plan, cutter)
    pmat, nodes, tex = plate_material(plan, work, plate, developer)
    plate.data.materials.append(pmat)
    if tex.get("uv_emit"):
        plate.lightgroup = "UV"

    dev_mat = developer_material(developer) if developer is not None else None
    grain = grain_of(plan["plate"].get("material", {}))    # sheets without their own grain follow the plate
    build_sheets(plan, cutter, lambda s: dev_mat or simple_material(f"sheet_{s['id']}", s.get("material", {}),
                                                                   grain_deg=grain))
    if cutter is not None:
        bpy.data.objects.remove(cutter)

    by_id = {h["id"]: h for h in holes}
    mats = {"mandrel": dev_mat or simple_material("mandrel", MANDREL)}
    for rv in plan.get("rivets") or []:
        hole = by_id.get(rv.get("hole"))
        if hole is None:
            meta["warnings"].append(f"rivet {rv.get('id')}: hole {rv.get('hole')} not in plan, skipped")
            continue
        spec = rv.get("material") or (STEEL if rv.get("kind") == "screw" else {})
        m = dev_mat or simple_material(f"rivet_{rv['id']}", spec)
        build_rivet(rv, hole, plan, dict(mats, rivet=m, screw=m))

    build_background(plan)
    build_gloves(plan)
    build_device_box(cam)
    build_hood(plan, cam)
    build_lights(plan)
    build_world(plan, meta)
    meta["K_render"] = build_camera(plan, cam, args.res_scale)
    return nodes, tex


# ---------------------------------------------------------------- overview (demo picture of the twin)

def emission_material(name: str, color, strength: float):
    mat, nt, out = new_material(name)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgba(color, (1.0, 1.0, 1.0))
    em.inputs["Strength"].default_value = float(strength)
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def camera_only(obj: bpy.types.Object) -> bpy.types.Object:
    """Seen by camera rays (and reflections) only: lights nothing and casts no shadow."""
    obj.visible_diffuse = False
    obj.visible_shadow = False
    obj.visible_transmission = False
    obj.visible_volume_scatter = False
    return obj


def led_lux(plan: dict, cam: dict) -> float:
    """Analytic white-LED illuminance (lux) at the optical-axis plate point (as lighting.led_irradiance)."""
    M = Matrix(cam["cv2world"])
    C, a = M.translation, M.to_3x3() @ Vector((0.0, 0.0, 1.0))
    if a.z > -1e-9:
        return 0.0
    p = C + (-C.z / a.z) * a
    e = 0.0
    for L in plan.get("lights") or []:
        if L.get("group") not in ("L", "R"):
            continue
        v = Vector(L["pos_m"]) - p
        d = v.length
        if v.z <= 0:
            continue
        mask = 1.0
        if str(L.get("kind", "spot")).lower() == "spot":
            ch = math.cos(math.radians(min(float(L.get("spot_deg", 180.0)), 180.0)) / 2)
            ca = Vector(L["dir"]).normalized().dot(-v / d)
            blend = float(L.get("blend", 0.0))
            f = 0.0 if ca <= ch else (1.0 if blend <= 0 else min(1.0, (ca - ch) / ((1 - ch) * blend)))
            mask = f * f * (3 - 2 * f)
        e += float(L["power_w"]) / (4 * math.pi * d * d) * (v.z / d) * mask
    return e * LM_PER_W


def add_pod_glow(plan: dict, cam: dict) -> None:
    """A glowing slab over each white LED pod (camera-frame bbox of its LEDs) at the pod's real radiance.
    Ring LEDs (mount 'ring', inside the opaque hood) get none: their bbox spans the hood interior, and
    the slab drew the whole hood as a lit white box."""
    M = Matrix(cam["cv2world"])
    Mi = M.inverted()
    for g in ("L", "R"):
        leds = [L for L in plan.get("lights") or [] if L.get("group") == g and L.get("mount") != "ring"]
        if not leds:
            continue
        pts = [Mi @ Vector(L["pos_m"]) for L in leds]
        x0, x1 = min(p.x for p in pts) - 0.006, max(p.x for p in pts) + 0.006
        y0, y1 = min(p.y for p in pts) - 0.006, max(p.y for p in pts) + 0.006
        bm = bmesh.new()
        add_box(bm, x0, x1, y0, y1, 0.0, max(p.z for p in pts) + 0.002)
        obj = mesh_object(f"pod_glow_{g}", bm)
        obj.matrix_world = M
        flux_w = sum(float(L["power_w"]) for L in leds) / 4.0        # Lambertian LED: P = 4 * flux / 683
        obj.data.materials.append(emission_material(f"pod_{g}", leds[0].get("color"), flux_w / (math.pi * (x1 - x0) * (y1 - y0))))
        camera_only(obj)


def add_lamp_proxies(amb: dict) -> None:
    """Emissive stand-ins so the overview shows the bulbs, lamps, panels and windows (they light nothing)."""
    for i, L in enumerate(amb.get("lights") or []):
        kind = str(L.get("kind", "")).lower()
        if kind in ("point", "spot"):
            r = max(float(L.get("radius_m", 0.01)), 0.01)
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=r)
            obj = mesh_object(f"lamp_proxy_{i}", bm)
            obj.location = tuple(float(x) for x in L["pos_m"])
            radiance = float(L["power_w"]) / (4 * math.pi) / (math.pi * r * r)
        elif kind == "area":
            w, h = (float(x) for x in (list(L.get("size_m") or [0.1]) * 2)[:2])
            if str(L.get("shape", "RECTANGLE")).upper() == "DISK":
                h = w
            bm = bmesh.new()
            bm.faces.new([bm.verts.new((a * w / 2, b * h / 2, 0.0)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            obj = mesh_object(f"lamp_proxy_{i}", bm)
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = light_quaternion(L)
            obj.location = Vector(L["pos_m"]) - 0.002 * Vector(L["dir"]).normalized()   # just behind the light
            radiance = float(L["power_w"]) / (math.pi * w * h)
        else:
            continue
        obj.data.materials.append(emission_material(f"lamp_proxy_{i}", L.get("color"), radiance))
        camera_only(obj)


def overview_camera(plan: dict, cam: dict) -> None:
    """Third-person camera OVERVIEW_DIST_M from a point between plate and device, on the side away from the main lamp."""
    W, H = (float(x) / 1000.0 for x in plan["plate"]["size_mm"])
    M = Matrix(cam["cv2world"])
    pc = Vector((W / 2, -H / 2, 0.0))
    target = pc.lerp(M.translation, 0.5)
    ycam = M.to_3x3() @ Vector((0.0, 1.0, 0.0))
    side = Vector((ycam.x, ycam.y, 0.0))
    side = side.normalized() if side.length > 1e-6 else Vector((0.0, -1.0, 0.0))
    lights = (plan.get("ambient") or {}).get("lights") or []
    if lights:
        main = max(lights, key=lambda L: float(L.get("lux", 0.0)))
        hl = -Vector(main["dir"]) if main.get("kind") == "sun" else Vector(main["pos_m"]) - pc
        hl.z = 0.0
        if hl.length > 1e-6 and side.dot(hl) > 0:
            side = -side                                   # look towards the lamp: it is in the picture
    side = Matrix.Rotation(math.radians(30.0), 3, "Z") @ side
    el = math.radians(OVERVIEW_ELEV_DEG)
    pos = target + OVERVIEW_DIST_M * (math.cos(el) * side + Vector((0.0, 0.0, math.sin(el))))
    cd = bpy.data.cameras.new("overview")
    cd.lens, cd.sensor_width, cd.sensor_fit = 30.0, 36.0, "HORIZONTAL"
    cd.clip_start, cd.clip_end = 0.01, 50.0
    obj = link(bpy.data.objects.new("overview", cd))
    obj.location = pos
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (target - pos).to_track_quat("-Z", "Y")
    bpy.context.scene.camera = obj


def render_overview(plan: dict, cam: dict, work: Path, args, meta: dict) -> None:
    """overview.png: the scene as built, seen from outside (device shell, glowing pods, lamps), denoised.

    Exposure follows the light on the plate (ambient lux + LED lux), so each condition reads as
    a photographer would expose it; a failure here is a warning, never the scene's error.
    """
    try:
        scn = bpy.context.scene
        dev = bpy.data.objects.get("device_box")
        if dev is not None:
            dev.visible_camera = True
            dev.data.materials.clear()
            dev.data.materials.append(simple_material("device_shell", {"base_color": [0.07, 0.07, 0.075],
                                                                        "roughness": 0.45}, metallic=0.0))
        add_pod_glow(plan, cam)
        amb = plan.get("ambient") or {}
        add_lamp_proxies(amb)
        overview_camera(plan, cam)
        scn.render.resolution_x, scn.render.resolution_y = OVERVIEW_RES
        scn.render.resolution_percentage = 100
        cy = scn.cycles
        cy.samples = int(args.overview_samples)
        cy.use_adaptive_sampling, cy.adaptive_threshold = True, 0.05
        cy.use_denoising = True
        try:
            scn.view_settings.view_transform = "AgX"
        except TypeError:
            scn.view_settings.view_transform = "Standard"
        lux = float(amb.get("lux") or 0.0) + led_lux(plan, cam)
        e_w = max(lux, 20.0) / LM_PER_W
        scn.view_settings.exposure = math.log2(0.18 * math.pi / (0.5 * e_w))   # albedo 0.5 under e_w -> 0.18
        meta["overview_exposure_ev"] = round(scn.view_settings.exposure, 3)
        s = scn.render.image_settings
        if hasattr(s, "media_type"):
            s.media_type = "IMAGE"
        s.file_format, s.color_mode, s.color_depth = "PNG", "RGB", "8"
        render_to(work / "overview.png", meta)
    except Exception as e:                                 # noqa: BLE001  (demo picture only)
        traceback.print_exc()
        meta["warnings"].append(f"overview failed: {e}")


def run_scene(work: Path, args) -> dict:
    meta = {"scene_id": work.name, "blender_version": bpy.app.version_string, "device_requested": args.device,
            "device": None, "res_scale": args.res_scale, "errors": [], "warnings": [], "files": []}
    t0 = time.time()
    try:
        with log_to(work / "blender.log"):
            try:
                plan = json.loads((work / "plan.json").read_text())
                cam = json.loads((work / "blender_cam.json").read_text())
                meta["scene_id"] = plan.get("scene_id", work.name)
                bpy.ops.wm.read_factory_settings(use_empty=True)
                scn = bpy.context.scene
                scn.render.engine = "CYCLES"
                meta["device"] = setup_device(args.device)
                meta["samples"] = setup_render(plan, args.samples)
                nodes, tex = build_scene(plan, cam, work, args, meta)
                meta["resolution"] = [scn.render.resolution_x, scn.render.resolution_y]
                meta["roi_size"] = [int(x) for x in cam["resolution"]]
                meta["seconds_build"] = round(time.time() - t0, 2)
                t1 = time.time()
                render_lit(work / "render.exr", plan, meta)
                meta["seconds_render"] = round(time.time() - t1, 2)
                if (plan.get("render") or {}).get("clean_pass"):
                    t2 = time.time()
                    swap_to_clean(nodes, tex, work, meta)
                    render_lit(work / "render_clean.exr", plan, meta)
                    meta["seconds_clean"] = round(time.time() - t2, 2)
                if getattr(args, "overview", False):
                    t3 = time.time()
                    render_overview(plan, cam, work, args, meta)
                    meta["seconds_overview"] = round(time.time() - t3, 2)
            except Exception:
                traceback.print_exc()
                raise
    except Exception:
        meta["errors"].append(traceback.format_exc())
    meta["error"] = meta["errors"][0].strip().splitlines()[-1] if meta["errors"] else None
    meta["seconds"] = round(time.time() - t0, 2)
    try:
        (work / "render_meta.json").write_text(json.dumps(meta, indent=1))
    except OSError as e:
        print(f"cannot write meta for {work}: {e}")
    return meta


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    for d in args.scenes:
        meta = run_scene(Path(d).resolve(), args)
        short = {k: meta.get(k) for k in ("scene_id", "device", "resolution", "seconds", "error")}
        print("AYAS_SCENE " + json.dumps(short), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
