# Copy the AICopilot addon from this checkout into FreeCAD's user Mod folder
# on Windows (the PowerShell counterpart of deploy.sh's addon step).
#
#   powershell -File scripts/deploy_windows.ps1 [-FreeCADBin <path to freecadcmd.exe>]
#
# The Mod folder is asked from FreeCAD itself (it is version-stamped, e.g.
# %APPDATA%\FreeCAD\v1-1\Mod). Restart FreeCAD afterwards, or call
# reload_modules for handler-only changes.
param(
    [string]$FreeCADBin = "C:\Program Files\FreeCAD 1.1\bin\freecadcmd.exe"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$line = & $FreeCADBin -c "import FreeCAD, os; print('MODDIR=' + os.path.join(FreeCAD.getUserAppDataDir(), 'Mod'))" 2>$null |
    Select-String '^MODDIR=' | Select-Object -Last 1
if (-not $line) { throw "Could not ask $FreeCADBin for its Mod folder" }
$mod = $line.ToString().Substring(7).Trim()
$target = Join-Path $mod "AICopilot"
New-Item -ItemType Directory -Force $mod | Out-Null
if (Test-Path $target) { Remove-Item -Recurse -Force $target }
Copy-Item -Recurse (Join-Path $repo "AICopilot") $target
Get-ChildItem $target -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Write-Output "AICopilot deployed to $target"
