# Per-session environment for PowerShell: dot-source it with  . .\env.ps1
# Keeps temp files and caches inside the project folder and activates the project's virtual environment.
$root = $PSScriptRoot
New-Item -ItemType Directory -Force "$root\tmp", "$root\cache\pip", "$root\cache\hf", "$root\cache\torch" | Out-Null
$env:TMP = "$root\tmp"; $env:TEMP = "$root\tmp"
$env:PIP_CACHE_DIR = "$root\cache\pip"
$env:HF_HOME = "$root\cache\hf"
$env:TORCH_HOME = "$root\cache\torch"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"
. "$root\.venv\Scripts\Activate.ps1"
