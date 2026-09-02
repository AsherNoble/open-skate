"""MJX (GPU-batched) backend for Open Skate.

The CPU path in `opensk.sim` stays the reference implementation and the source
of truth for geometry: both backends build from the same MJCF and the same
`SkateParams`.
"""
