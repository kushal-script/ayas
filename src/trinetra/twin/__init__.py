"""Digital twin of the TriNetra handheld and the test coupons, for synthetic pretraining.

See docs/twin/DESIGN.md for the frames, the scene-plan format and the data flow:
plan (sampler) -> textures (numpy) -> render (Blender/Cycles) -> post (labels, tiles).
"""
