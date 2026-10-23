# About this fork

This is my fork of [blwfish/freecad-mcp](https://github.com/blwfish/freecad-mcp).
I use it as the CAD tool for a design assistant I'm building, on Windows 11
with FreeCAD 1.1. I needed two things upstream didn't have: FEM, and things
that work properly on Windows. Everything else is upstream as is.

## What's different

**FEM (`fem_operations`).** Static structural analysis with Gmsh and
CalculiX: create an analysis, set a material, mesh, fix faces, add a force,
pressure or gravity, run, and get back max von Mises stress, max
displacement and a safety factor against yield. The solver part is based on
[neka-nat/freecad-mcp](https://github.com/neka-nat/freecad-mcp) (MIT, see
`LICENSES/`). The mesh is second order by default, because first-order
tetrahedra came out about a third too stiff on a simple cantilever.

**Windows fixes.**
- Headless instances (`spawn_freecad_instance`) now start on Windows.
- A long task (like a solve) no longer makes the next call fail with
  "GUI thread appears unresponsive".
- The crash log is written on Windows too.

**Smaller fixes.**
- `save_fixture` works on boolean parts.
- Fixtures no longer end up inside FreeCAD's Mod folder.
- `delete_object` can be undone, and `undo` tells you what it did.
- `rollback_to_checkpoint` says which deleted objects it can't bring back.

## Installing on Windows

```powershell
git clone https://github.com/omiata/freecad-mcp-ppc.git
cd freecad-mcp-ppc
uv sync
powershell -File scripts/deploy_windows.ps1
claude mcp add freecad -e PYTHONUTF8=1 -- "<repo>\.venv\Scripts\python.exe" "<repo>\freecad_mcp_server.py"
```

Restart FreeCAD afterwards. `PYTHONUTF8=1` is there because the bridge
prints emoji, which the Windows console can't handle otherwise.

## Keeping up with upstream

```bash
git fetch upstream
git merge upstream/main
```

## Things to know

- About 170 unit tests fail on Windows. They fail the same way on upstream:
  they test Unix-only things (Unix sockets, `/tmp`, process groups). Compare
  against upstream rather than expecting a clean run.
- FreeCAD freezes while CalculiX runs. For normal parts that's a second or
  two.
- Only static analysis for now: no thermal, modal or buckling.
