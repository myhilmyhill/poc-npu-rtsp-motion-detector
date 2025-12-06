#requires -Version 5.1
[CmdletBinding()] param(
    [string]$ModelPath = "models\yolov8n.onnx",
    [string]$OutputDir = "models\snpe"
)

$ErrorActionPreference = 'Stop'

# Check for micromamba
$mm = ".\.mamba\micromamba.exe"
if (!(Test-Path $mm)) {
    Write-Error "Micromambaが見つかりません。まず .\scripts\setup_windows_portable.ps1 を実行してください。"
}

Write-Host "`n===== YOLOv8 → SNPE DLC 変換 =====`n" -ForegroundColor Cyan

# Check SNPE_ROOT
if (-not $env:SNPE_ROOT) {
    Write-Error "SNPE_ROOT環境変数が設定されていません"
}

# Detect Python Architecture
$pythonArchScript = "import platform; print(platform.machine())"
$pythonArch = & $mm run -p .\.venv-mamba python -c $pythonArchScript 2>$null
Write-Host "Python Architecture: $pythonArch" -ForegroundColor Cyan

$targetArch = "ARM64"
$binArch = "arm64x-windows-msvc"

if ($pythonArch -match "AMD64|x86_64") {
    $targetArch = "X86_64"
    $binArch = "x86_64-windows-msvc"
} elseif ($pythonArch -match "ARM64|aarch64") {
    $targetArch = "ARM64"
    $binArch = "arm64x-windows-msvc"
} else {
    # Fallback to system architecture
    $isArm64 = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq 'Arm64'
    if (-not $isArm64) {
        $targetArch = "X86_64"
        $binArch = "x86_64-windows-msvc"
    }
}

Write-Host "Target QAIRT Architecture: $targetArch ($binArch)" -ForegroundColor Cyan

# Run QAIRT environment setup script
$envSetup = Join-Path $env:SNPE_ROOT "bin\envsetup.ps1"
if (Test-Path $envSetup) {
    Write-Host "QAIRT環境をセットアップ中 (-arch $targetArch)..." -ForegroundColor Cyan
    Unblock-File $envSetup -ErrorAction SilentlyContinue
    . $envSetup -arch $targetArch
}

# FIX: Add platform specific python path for libPyIrGraph
$commonPath = Join-Path $env:SNPE_ROOT "lib\python\qti\aisw\converters\common"
if ($targetArch -eq "ARM64") {
    $libPath = Join-Path $commonPath "windows-arm64ec"
} else {
    $libPath = Join-Path $commonPath "windows-x86_64"
}

if (Test-Path $libPath) {
    Write-Host "Adding to PYTHONPATH: $libPath" -ForegroundColor Cyan
    $env:PYTHONPATH = "$libPath;$env:PYTHONPATH"
    
    # Also add to PATH for DLL dependencies between pyd files
    Write-Host "Adding to PATH: $libPath" -ForegroundColor Cyan
    $env:PATH = "$libPath;$env:PATH"
}

# Find converter tool
$qnnTool = Join-Path $env:SNPE_ROOT "bin\$binArch\qnn-onnx-converter"
if (Test-Path $qnnTool) {
    $tool = $qnnTool
    Write-Host "ツール: qnn-onnx-converter" -ForegroundColor Green
} else {
    Write-Error "qnn-onnx-converterが見つかりません: $qnnTool"
}

# Check model
if (!(Test-Path $ModelPath)) {
    Write-Error "モデルが見つかりません: $ModelPath"
}

# Create output dir
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$dlcPath = Join-Path $OutputDir "yolov8n.dlc"
Write-Host "入力: $ModelPath" -ForegroundColor White
Write-Host "出力: $dlcPath" -ForegroundColor White
Write-Host "`n変換中 (micromamba Python使用)..." -ForegroundColor Yellow

# Debug: Check environment
Write-Host "`n=== Environment Debug ===" -ForegroundColor Yellow
Write-Host "PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Gray
Write-Host "PATH (first 500 chars): $($env:PATH.Substring(0, [Math]::Min(500, $env:PATH.Length)))" -ForegroundColor Gray
Write-Host "Tool: $tool" -ForegroundColor Gray

& $mm run -p .\.venv-mamba python debug_env.py

Write-Host "`n=== Attempting Conversion ===" -ForegroundColor Yellow
# Use micromamba Python with QAIRT environment variables inherited
# micromamba run will pass through the current environment (PATH, PYTHONPATH, etc.)
& $mm run -p .\.venv-mamba python $tool --input_network $ModelPath --output_path $dlcPath --input_layout "images" "NCHW" --input_dim "images" "1,3,640,640"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ 変換成功！" -ForegroundColor Green
    $fileSize = (Get-Item $dlcPath).Length / 1MB
    Write-Host "ファイルサイズ: $([math]::Round($fileSize, 2)) MB`n" -ForegroundColor Gray
} else {
    Write-Error "変換失敗 (終了コード: $LASTEXITCODE)"
}
