# Saved initial AMReX candidates

These three D2Q9 L0–L2 snapshots are generated initial BoxArrays before project gap/nesting repair. Coordinates are integers with exclusive upper endpoints. Each row is `level ilo ihi jlo jhi klo khi`; original FAB order is retained.

The fixture geometry uses L0 `128×128×1`, refinement ratio `(2,2,1)`, grid efficiency `0.7`, maximum grid size `(32,32,1)`, gap/nesting `4/4` and blocking `(1,1,1)`. The samples at steps 832, 1280 and 1664 represent `t/tc=0.65,1.0,1.3`. The source CSV digest is `86ce3830c325f1fb7e7d45c58986db98a3cd5ef75f09b22763234668d120622d`.

These are geometry inputs, not field data or repaired-output expectations. Tests independently check preservation of original coverage and final geometry constraints. The 832 sample must retain its original BoxArray exactly. Candidate files written under the test-owned `artifacts/dynamic_clustering/` support shape comparison; they are not a scientific accuracy or performance reference.
