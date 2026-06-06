#requires -Version 5.1
<#
.SYNOPSIS
    One-shot installer for image2video on Windows.

.DESCRIPTION
    Creates a conda env, clones ComfyUI + the custom nodes Sulphur-2
    needs, downloads the Sulphur-2 FP8 checkpoint + Qwen prompt
    enhancer, copies the bundled Sulphur workflows into ./workflows/,
    and patches ComfyUI's startup so it finds cuDNN DLLs on Windows.

.PARAMETER Force
    Re-run download / patch steps even if targets exist. Useful after
    bumping model versions or upstream commit pins.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Force          # re-pull everything

.NOTES
    Targets ~30 min wall-clock on a fast connection. Total ~80 GB on
    disk. Prereqs: conda + git + ~80 GB free.
#>
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }

# ----------------------------------------------------------------------
# Versions / sources — pin so this script keeps working as upstreams move.
# ----------------------------------------------------------------------
$EnvName        = "img2vid"
$PyVersion      = "3.11"
$ComfyUiRepo    = "https://github.com/comfyanonymous/ComfyUI.git"
$ComfyMgrRepo   = "https://github.com/ltdrdata/ComfyUI-Manager.git"
$LTXVideoRepo   = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
$GGUFNodeRepo   = "https://github.com/city96/ComfyUI-GGUF.git"
$PromptRelayRepo= "https://github.com/kijai/ComfyUI-PromptRelay.git"

# Sulphur-2-base ships several checkpoints; we pull the FP8 variant
# (~29 GB) which is the sweet spot for 24 GB VRAM. See model card.
$SulphurRepo    = "SulphurAI/Sulphur-2-base"
$SulphurCkpt    = "sulphur_dev_fp8mixed.safetensors"

# Qwen prompt enhancer. 7B-Instruct in 4-bit needs ~5 GB VRAM; loads
# only while enhancing a prompt and unloads afterwards.
$QwenRepo       = "Qwen/Qwen2.5-7B-Instruct"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
function Write-Step($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

function Assert-Exe($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "$name not found on PATH. $hint"
    }
}

function Run-Conda([string[]]$argList) {
    # conda emits warnings on stderr that PS5 wraps as ErrorRecords —
    # capture both into stdout to avoid the cmdlet-error tarpit.
    & conda @argList 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "conda $($argList -join ' ') failed (exit $LASTEXITCODE)" }
}

function Clone-Or-Update($url, $dir) {
    if (Test-Path $dir) {
        if ($Force) {
            Write-Host "Updating $dir"
            git -C $dir fetch --depth 1 origin
            git -C $dir reset --hard origin/HEAD
        } else {
            Write-Host "Already cloned: $dir (skipping; use -Force to update)"
        }
    } else {
        Write-Host "Cloning $url -> $dir"
        git clone --depth 1 $url $dir
        if ($LASTEXITCODE -ne 0) { throw "git clone $url failed" }
    }
}

# ----------------------------------------------------------------------
# 0. Prereqs
# ----------------------------------------------------------------------
Write-Step "0/7  Checking prerequisites"
Assert-Exe "conda" "Install Anaconda or Miniconda: https://www.anaconda.com/download"
Assert-Exe "git"   "Install Git for Windows: https://git-scm.com/download/win"

# ----------------------------------------------------------------------
# 1. Conda env
# ----------------------------------------------------------------------
Write-Step "1/7  Conda env '$EnvName' (Python $PyVersion)"
$envList = (& conda env list) -join "`n"
if ($envList -match "(?m)^\s*$EnvName\s") {
    Write-Host "Env exists. Use 'conda env remove -n $EnvName' to rebuild."
} else {
    Run-Conda @("create","-y","-n",$EnvName,"python=$PyVersion","pip")
}

# ----------------------------------------------------------------------
# 2. Clone ComfyUI + custom nodes
# ----------------------------------------------------------------------
Write-Step "2/7  Cloning ComfyUI + custom nodes"
$ComfyDir   = Join-Path $RepoRoot "comfyui"
$CustomDir  = Join-Path $ComfyDir "custom_nodes"

Clone-Or-Update $ComfyUiRepo $ComfyDir
New-Item -ItemType Directory -Force -Path $CustomDir | Out-Null
Clone-Or-Update $ComfyMgrRepo    (Join-Path $CustomDir "ComfyUI-Manager")
Clone-Or-Update $LTXVideoRepo    (Join-Path $CustomDir "ComfyUI-LTXVideo")
Clone-Or-Update $GGUFNodeRepo    (Join-Path $CustomDir "ComfyUI-GGUF")
Clone-Or-Update $PromptRelayRepo (Join-Path $CustomDir "ComfyUI-PromptRelay")

# ----------------------------------------------------------------------
# 3. Install Python deps into the conda env
# ----------------------------------------------------------------------
Write-Step "3/7  Installing Python deps into '$EnvName'"

# 3a. ComfyUI's own requirements (torch + cu121 + xformers + etc.)
$ComfyReq = Join-Path $ComfyDir "requirements.txt"
Run-Conda @("run","-n",$EnvName,"--no-capture-output",
            "pip","install","-r",$ComfyReq)

# 3b. Each custom node's requirements.txt (Manager, LTXVideo, etc.)
foreach ($nodeDir in (Get-ChildItem $CustomDir -Directory)) {
    $req = Join-Path $nodeDir.FullName "requirements.txt"
    if (Test-Path $req) {
        Write-Host "Installing deps for $($nodeDir.Name)"
        Run-Conda @("run","-n",$EnvName,"--no-capture-output",
                    "pip","install","-r",$req)
    }
}

# 3c. Our own webapp deps
$WebReq = Join-Path $RepoRoot "requirements.txt"
Run-Conda @("run","-n",$EnvName,"--no-capture-output",
            "pip","install","-r",$WebReq)

# ----------------------------------------------------------------------
# 4. Download Sulphur-2 checkpoint + Qwen prompt enhancer
# ----------------------------------------------------------------------
Write-Step "4/7  Downloading models (~50 GB; this can take a while)"

$CkptDir = Join-Path $ComfyDir "models\checkpoints"
New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null

$ModelsCache = Join-Path $RepoRoot "models"
New-Item -ItemType Directory -Force -Path $ModelsCache | Out-Null

# Use huggingface_hub.snapshot_download via python so we get resume,
# parallel chunks, and proper auth handling.
$DownloadPy = @"
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download

repo_root  = r'$RepoRoot'.replace('\\\\','/')
ckpt_dir   = r'$CkptDir'.replace('\\\\','/')
models_cache = r'$ModelsCache'.replace('\\\\','/')

# Sulphur-2 FP8 checkpoint -> ComfyUI/models/checkpoints/
ckpt = hf_hub_download(
    repo_id='$SulphurRepo',
    filename='$SulphurCkpt',
    local_dir=ckpt_dir,
    local_dir_use_symlinks=False,
)
print('Downloaded:', ckpt)

# Sulphur's bundled workflows -> ./workflows/
snapshot_download(
    repo_id='$SulphurRepo',
    allow_patterns=['workflows/*'],
    local_dir=os.path.join(models_cache, 'sulphur'),
    local_dir_use_symlinks=False,
)
print('Downloaded Sulphur workflows')

# Qwen2.5-7B-Instruct (full repo for transformers.from_pretrained)
snapshot_download(
    repo_id='$QwenRepo',
    local_dir=os.path.join(models_cache, 'qwen2.5-7b-instruct'),
    local_dir_use_symlinks=False,
)
print('Downloaded Qwen2.5-7B-Instruct')
"@

$DownloadPyFile = Join-Path $env:TEMP "img2vid_download.py"
Set-Content -Path $DownloadPyFile -Value $DownloadPy -Encoding utf8
Run-Conda @("run","-n",$EnvName,"--no-capture-output","python",$DownloadPyFile)
Remove-Item $DownloadPyFile -ErrorAction SilentlyContinue

# Copy Sulphur's bundled workflows into ./workflows/ for our webapp.
$WfSrc = Join-Path $ModelsCache "sulphur\workflows"
$WfDst = Join-Path $RepoRoot "workflows"
if (Test-Path $WfSrc) {
    Write-Host "Copying Sulphur workflows -> ./workflows/"
    Copy-Item -Recurse -Force "$WfSrc\*" $WfDst
}

# ----------------------------------------------------------------------
# 5. cuDNN DLL discovery patch (Windows-only)
# ----------------------------------------------------------------------
# Python 3.8+ on Windows ignores PATH for native imports — onnxruntime /
# torch wheels that depend on nvidia-cudnn-cu12 etc. need an explicit
# os.add_dll_directory() call. Patch ComfyUI's entry point so the env
# is correct before torch is imported.
Write-Step "5/7  Applying cuDNN DLL-discovery patch to ComfyUI"

$PatchMarker = "# IMG2VID_CUDNN_PATCH"
$MainPy = Join-Path $ComfyDir "main.py"
$MainTxt = Get-Content $MainPy -Raw
if ($MainTxt -match $PatchMarker) {
    Write-Host "Already patched."
} else {
    $patch = @"
$PatchMarker
import os, sys
if sys.platform == 'win32':
    _sp = os.path.join(sys.prefix, 'Lib', 'site-packages')
    _cookies = []
    for _sub in ('cudnn','cublas','cuda_runtime','curand','cufft',
                 'cuda_nvrtc','nvjitlink'):
        _bin = os.path.join(_sp, 'nvidia', _sub, 'bin')
        if os.path.isdir(_bin):
            _cookies.append(os.add_dll_directory(_bin))
            os.environ['PATH'] = _bin + os.pathsep + os.environ['PATH']
    _trt = os.path.join(_sp, 'tensorrt_libs')
    if os.path.isdir(_trt):
        _cookies.append(os.add_dll_directory(_trt))
    # Keep cookies alive — module-level list is enough.
    globals()['_img2vid_dll_cookies'] = _cookies
"@
    Set-Content -Path $MainPy -Value ($patch + "`n" + $MainTxt) -Encoding utf8
    Write-Host "Patched $MainPy"
}

# ----------------------------------------------------------------------
# 6. Smoke test — start ComfyUI in the background, hit /system_stats
# ----------------------------------------------------------------------
Write-Step "6/7  Smoke test: launching ComfyUI headless on 127.0.0.1:8188"

$LogPath = Join-Path $RepoRoot "out\comfy_smoke.log"
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null

$proc = Start-Process -PassThru -WindowStyle Hidden -FilePath "conda" `
    -ArgumentList @("run","-n",$EnvName,"--no-capture-output","python",
                    (Join-Path $ComfyDir "main.py"),
                    "--listen","127.0.0.1","--port","8188") `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError  ($LogPath + ".err")

Write-Host "ComfyUI pid=$($proc.Id). Waiting up to 120 s for it to bind 8188..."
$ok = $false
for ($i=0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8188/system_stats" `
            -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
try { Stop-Process -Id $proc.Id -Force } catch { }

if ($ok) {
    Write-Host "ComfyUI bound 8188 OK — install is healthy." -ForegroundColor Green
} else {
    Write-Warning "ComfyUI did not bind 8188 within 120 s. See $LogPath / $LogPath.err"
    Write-Warning "Common causes: model still loading (huge checkpoint), port in use, missing CUDA."
}

# ----------------------------------------------------------------------
# 7. Done
# ----------------------------------------------------------------------
Write-Step "7/7  Done"
Write-Host ""
Write-Host "Start the web app with:" -ForegroundColor Green
Write-Host "    conda run -n $EnvName python webapp.py"
Write-Host ""
Write-Host "Then open  http://localhost:8080/"
