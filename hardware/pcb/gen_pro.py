"""Write the KiCad project file (.kicad_pro) with the fab design rules (Lion Circuits standard class)."""
import json, os, sys

RULES = {"min_clearance": 0.15, "min_copper_edge_clearance": 0.3, "min_hole_clearance": 0.18,
         "min_hole_to_hole": 0.25, "min_track_width": 0.125, "min_via_diameter": 0.45, "min_via_annular_width": 0.1,
         "min_through_hole_diameter": 0.25, "min_silk_clearance": 0.0}
SEVERITY = {"silk_over_copper": "warning", "silk_overlap": "warning", "silk_edge_clearance": "warning",
            "lib_footprint_issues": "ignore", "lib_footprint_mismatch": "ignore"}
CLASSES = [{"name": "Default", "clearance": 0.15, "track_width": 0.25, "via_diameter": 0.6, "via_drill": 0.3,
            "diff_pair_width": 0.2, "diff_pair_gap": 0.2, "wire_width": 6, "bus_width": 12, "line_style": 0,
            "schematic_color": "rgba(0, 0, 0, 0.000)", "pcb_color": "rgba(0, 0, 0, 0.000)", "priority": 2147483647},
           {"name": "Power", "clearance": 0.15, "track_width": 0.5, "via_diameter": 0.8, "via_drill": 0.4,
            "diff_pair_width": 0.2, "diff_pair_gap": 0.2, "wire_width": 6, "bus_width": 12, "line_style": 0,
            "schematic_color": "rgba(0, 0, 0, 0.000)", "pcb_color": "rgba(0, 0, 0, 0.000)", "priority": 0}]
POWER = ["VBUS", "V12", "V12_SW", "V12_LED", "V5", "SW", "GND"] + ["K_" + n for n in
         ("WHITE_L", "WHITE_R", "UV_L", "UV_R", "RING1", "RING2", "RING3", "RING4", "RING5", "RING6")] + \
        ["K_WHITE", "K_UV"] + ["S_" + n for n in ("WHITE_L", "WHITE_R", "UV_L", "UV_R", "RING1", "RING2", "RING3", "RING4", "RING5", "RING6")]


# LIGHT/IO (6-layer): finer rules so the RP2040's 0.4 mm-pitch QFN-56 can fan out: 0.1 mm track and space,
# 0.2 mm drill / 0.45 mm vias (a standard 6-layer capability; confirm with the fab quote)
FINE = {"light"}
RULES_FINE = dict(RULES, min_clearance=0.1, min_track_width=0.09,  # 0.09: neck-down at two RP2040 pins
                  min_via_diameter=0.4, min_through_hole_diameter=0.2, min_hole_clearance=0.15, min_hole_to_hole=0.2)


def classes_for(board):
    if board not in FINE:
        return CLASSES
    c = [dict(x) for x in CLASSES]
    c[0].update(clearance=0.1, track_width=0.15, via_diameter=0.45, via_drill=0.2)
    c[1].update(clearance=0.1)                       # power keeps its 0.5 mm tracks; spacing as the rest of the board
    return c


def write(board, out_dir):
    pro = {"board": {"design_settings": {"rules": RULES_FINE if board in FINE else RULES, "rule_severities": SEVERITY}},
           "net_settings": {"classes": classes_for(board), "meta": {"version": 4},
                            "netclass_patterns": [{"netclass": "Power", "pattern": n} for n in POWER]},
           "meta": {"filename": "ayas_%s.kicad_pro" % board, "version": 3}}
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "fp-lib-table"), "w").write(
        '(fp_lib_table\n  (version 7)\n  (lib (name "ayas")(type "KiCad")(uri "${KIPRJMOD}/../../ayas.pretty")(options "")(descr "Ayas custom footprints"))\n)\n')
    # Wurth WA-SMSI nuts: the footprint's own solder ring sits round its own hole (by design, not a defect)
    open(os.path.join(out_dir, "ayas_%s.kicad_dru" % board), "w").write(
        '(version 1)\n(rule "smd nut ring"\n  (condition "A.memberOfFootprint(\'N*\') && B.memberOfFootprint(\'N*\')")\n'
        '  (constraint hole_clearance (min 0mm)))\n')
    path = os.path.join(out_dir, "ayas_%s.kicad_pro" % board)
    json.dump(pro, open(path, "w"), indent=2)
    return path


if __name__ == "__main__":
    b = sys.argv[1]
    print(write(b, sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "kicad", b)))
