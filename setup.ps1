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
# Corporate-cert SSL bypass for conda / pip / huggingface_hub.
# This box has a corporate-issued root cert that Python's bundled CA
# store doesn't trust; without this conda + pip fail with SSL_CERT
# verify errors on every package fetch. We trust the local box.
# ----------------------------------------------------------------------
$env:CONDA_SSL_VERIFY = "false"
$env:PIP_TRUSTED_HOST = "pypi.org files.pythonhosted.org pypi.python.org download.pytorch.org"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:CURL_CA_BUNDLE = ""
$env:REQUESTS_CA_BUNDLE = ""

# ----------------------------------------------------------------------
# Versions / sources - pin so this script keeps working as upstreams move.
# ----------------------------------------------------------------------
$EnvName        = "img2vid"
$PyVersion      = "3.11"
$ComfyUiRepo    = "https://github.com/comfyanonymous/ComfyUI.git"
$ComfyMgrRepo   = "https://github.com/ltdrdata/ComfyUI-Manager.git"
$LTXVideoRepo   = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
$GGUFNodeRepo   = "https://github.com/city96/ComfyUI-GGUF.git"
$PromptRelayRepo= "https://github.com/kijai/ComfyUI-PromptRelay.git"
# KJNodes is required for several utility nodes (PathchSageAttentionKJ
# etc.) referenced by Sulphur's distilled workflow. Without it ComfyUI
# 400s on submit with missing_node_type.
$KJNodesRepo    = "https://github.com/kijai/ComfyUI-KJNodes.git"

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
    # DO NOT use `2>&1` here. PS5 wraps stderr from native exes as
    # ErrorRecords (NativeCommandError), and combined with
    # $ErrorActionPreference=Stop that aborts on the FIRST stderr line
    # even when conda's actual exit code is 0. Let stderr flow to its
    # own stream; the outer Tee-Object on the caller side still
    # captures both streams.
    & conda @argList
    if ($LASTEXITCODE -ne 0) {
        throw "conda $($argList -join ' ') failed (exit $LASTEXITCODE)"
    }
}

function Run-Pip([string[]]$pipArgs) {
    # Pip via conda run -n <env>, with trusted-host flags so corporate
    # cert chains don't tank wheel downloads.
    $extra = @(
        "--trusted-host","pypi.org",
        "--trusted-host","files.pythonhosted.org",
        "--trusted-host","pypi.python.org",
        "--trusted-host","download.pytorch.org"
    )
    Run-Conda (@("run","-n",$EnvName,"--no-capture-output","pip") + $pipArgs + $extra)
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
Clone-Or-Update $KJNodesRepo     (Join-Path $CustomDir "ComfyUI-KJNodes")

# ----------------------------------------------------------------------
# 3. Install Python deps into the conda env
# ----------------------------------------------------------------------
Write-Step "3/7  Installing Python deps into '$EnvName'"

# 3a. ComfyUI's own requirements (torch + cu121 + xformers + etc.)
$ComfyReq = Join-Path $ComfyDir "requirements.txt"
Run-Pip @("install","-r",$ComfyReq)

# 3b. Each custom node's requirements.txt (Manager, LTXVideo, etc.)
foreach ($nodeDir in (Get-ChildItem $CustomDir -Directory)) {
    $req = Join-Path $nodeDir.FullName "requirements.txt"
    if (Test-Path $req) {
        Write-Host "Installing deps for $($nodeDir.Name)"
        Run-Pip @("install","-r",$req)
    }
}

# 3c. Our own webapp deps
$WebReq = Join-Path $RepoRoot "requirements.txt"
Run-Pip @("install","-r",$WebReq)

# 3d. truststore + pip-system-certs. The corporate cert chain isn't in
# Python's bundled certifi, so requests/urllib3 fail SSL verify on
# huggingface_hub downloads. truststore.inject_into_ssl() makes
# Python's ssl module read from the Windows cert store instead -
# which already trusts the corporate root.
Run-Pip @("install","truststore","pip-system-certs")

# 3e. Force CUDA torch wheels.
# ComfyUI's requirements.txt + custom node requirements pull plain
# 'torch' from PyPI's default index, which on Windows is the CPU-only
# wheel. Running the model on that fails with "Torch not compiled
# with CUDA enabled" at startup. Reinstall from PyTorch's cu121 index
# AFTER all other pip steps so nothing downgrades us back to CPU.
Run-Pip @("install","--upgrade","--force-reinstall",
         "--index-url","https://download.pytorch.org/whl/cu121",
         "torch","torchvision","torchaudio")

# 3f. Pin kornia to 0.7.x. ComfyUI-LTXVideo imports
# 'pad' from kornia.geometry.transform.pyramid which was removed in
# kornia 0.8.x. Without this pin, LTXVideo silently fails to load
# during ComfyUI startup and every video-generation submit returns
# missing_node_type for ResizeImageResolution / LTXVScheduler / etc.
Run-Pip @("install","kornia==0.7.4")

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

# Corporate-cert SSL bypass. Two-layer approach:
#  1) truststore.inject_into_ssl() teaches Python ssl to use the Windows
#     trust store (which DOES trust the corporate root cert). This works
#     for requests, urllib3, and huggingface_hub.
#  2) If truststore isn't installed, fall back to monkey-patching every
#     requests.Session to set verify=False. Less secure but unblocks
#     the install on locked-down boxes.
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
try:
    import truststore
    truststore.inject_into_ssl()
    print('[ssl] using truststore (Windows cert store)')
except ImportError:
    print('[ssl] truststore unavailable; falling back to verify=False')
    import ssl, urllib3, requests
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context
    _orig_send = requests.adapters.HTTPAdapter.send
    def _send(self, request, **kwargs):
        kwargs['verify'] = False
        return _orig_send(self, request, **kwargs)
    requests.adapters.HTTPAdapter.send = _send

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

# 4b. Stage-2 + stage-3 asset downloads. Sulphur's workflow references
# files that aren't in the main checkpoint repo - the LTX spatial
# upscaler (Lightricks/LTX-2.3), the Gemma-3 text encoder
# (inflatebot/...), Sulphur LoRAs (distill_loras/ subdir), plus a
# placeholder PNG for T2V LoadImage. Without these, ComfyUI 400s on
# /prompt with value_not_in_list for every loader. ~22 GB.
Write-Host ""
Write-Host "Stage 2: spatial upscaler + T2V placeholder image"
Run-Conda @("run","-n",$EnvName,"--no-capture-output","python",
            (Join-Path $RepoRoot "tools\download_stage2_assets.py"))
Write-Host ""
Write-Host "Stage 3: Gemma encoder + audio VAE + LoRAs (~22 GB; longest step)"
Run-Conda @("run","-n",$EnvName,"--no-capture-output","python",
            (Join-Path $RepoRoot "tools\download_stage3_assets.py"))

# ----------------------------------------------------------------------
# 5. cuDNN DLL discovery patch (Windows-only)
# ----------------------------------------------------------------------
# Python 3.8+ on Windows ignores PATH for native imports - onnxruntime /
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
    # Keep cookies alive - module-level list is enough.
    globals()['_img2vid_dll_cookies'] = _cookies
"@
    Set-Content -Path $MainPy -Value ($patch + "`n" + $MainTxt) -Encoding utf8
    Write-Host "Patched $MainPy"
}

# ----------------------------------------------------------------------
# 6. Smoke test - start ComfyUI in the background, hit /system_stats
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
    Write-Host "ComfyUI bound 8188 OK - install is healthy." -ForegroundColor Green
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
