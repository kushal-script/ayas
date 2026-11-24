"""Write the KiCad project file (.kicad_pro) with the fab design rules (Lion Circuits standard class)."""
import json, os, sys

RULES = {"min_clearance": 0.15, "min_copper_edge_clearance": 0.3, "min_hole_clearance": 0.18,
         "min_hole_to_hole": 0.25, "min_track_width": 0.15, "min_via_diameter": 0.45, "min_via_annular_width": 0.1,
         "min_through_hole_diameter": 0.25, "min_silk_clearance": 0.0}
SEVERITY = {"silk_over_copper": "warning", "silk_overlap": "warning", "silk_edge_clearance": "warning",
            "lib_footprint_issues": "ignore", "lib_footprint_mismatch": "ignore"}
CLASSES = [{"name": "Default", "clearance": 0.15, "track_width": 0.25, "via_diameter": 0.6, "via_drill": 0.3,
            "diff_pair_width": 0.2, "diff_pair_gap": 0.2, "wire_width": 6, "bus_width": 12, "line_style": 0,
            "schematic_color": "rgba(0, 0, 0, 0.000)", "pcb_color": "rgba(0, 0, 0, 0.000)", "priority": 2147483647},
           {"name": "Power", "clearance": 0.15, "track_width": 0.8, "via_diameter": 0.8, "via_drill": 0.4,
            "diff_pair_width": 0.2, "diff_pair_gap": 0.2, "wire_width": 6, "bus_width": 12, "line_style": 0,
            "schematic_color": "rgba(0, 0, 0, 0.000)", "pcb_color": "rgba(0, 0, 0, 0.000)", "priority": 0}]
POWER = ["VBUS", "V12", "V12_SW", "V12_LED", "V5", "SW", "GND"] + ["K_" + n for n in
         ("WHITE_L", "WHITE_R", "UV_L", "UV_R", "RING1", "RING2", "RING3", "RING4", "RING5", "RING6")] + \
        ["S_" + n for n in ("WHITE_L", "WHITE_R", "UV_L", "UV_R", "RING1", "RING2", "RING3", "RING4", "RING5", "RING6")]


def write(board, out_dir):
    pro = {"board": {"design_settings": {"rules": RULES, "rule_severities": SEVERITY}},
           "net_settings": {"classes": CLASSES, "meta": {"version": 4},
                            "netclass_patterns": [{"netclass": "Power", "pattern": n} for n in POWER]},
           "meta": {"filename": "ayas_%s.kicad_pro" % board, "version": 3}}
    path = os.path.join(out_dir, "ayas_%s.kicad_pro" % board)
    json.dump(pro, open(path, "w"), indent=2)
    return path


if __name__ == "__main__":
    b = sys.argv[1]
    print(write(b, sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "kicad", b)))
