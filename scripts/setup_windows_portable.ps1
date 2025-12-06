#requires -Version 5.1
<#
.SYNOPSIS
    NPU Object Detection Demo用のポータブルPython環境をセットアップ

.DESCRIPTION
    Micromambaを使用してローカルPython環境を作成し、
    ONNX Runtime QNNおよび必要なパッケージをインストールします。

.PARAMETER EnvPath
    作成する仮想環境のパス（デフォルト: .\.venv-mamba）

.PARAMETER MicromambaPath
    Micromambaの実行ファイルパス（デフォルト: .\.mamba\micromamba.exe）

.EXAMPLE
    .\scripts\setup_windows_portable.ps1
#>

[CmdletBinding()] param(
    [string]$EnvPath = ".\.venv-mamba",
    [string]$MicromambaPath = ".\.mamba\micromamba.exe"
)

$ErrorActionPreference = 'Stop'

Write-Host "`n===== NPU Object Detection Demo Setup =====`n" -ForegroundColor Cyan

# ===== Functions =====

function Download-File($Url, $Out) {
    Write-Host "[setup] Downloading $Url -> $Out" -ForegroundColor Cyan
    $dir = Split-Path -Parent $Out
    if (!(Test-Path $dir)) { 
        New-Item -ItemType Directory -Path $dir -Force | Out-Null 
    }
    Invoke-WebRequest -Uri $Url -OutFile $Out
}

function Ensure-Micromamba {
    Write-Host "[1/3] Micromambaの確認..." -ForegroundColor Yellow
    
    if (Test-Path $MicromambaPath) {
        Write-Host "  ✓ Micromamba already exists: $MicromambaPath" -ForegroundColor Green
        return
    }
    
    Write-Host "  Micromambaをダウンロード中..." -ForegroundColor Cyan
    $url = "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64.exe"
    Download-File $url $MicromambaPath
    Write-Host "  ✓ Micromambaダウンロード完了" -ForegroundColor Green
}

function Create-Env {
    Write-Host "`n[2/3] Python環境の作成..." -ForegroundColor Yellow
    
    # Ensure local micromamba root exists
    $root = Join-Path -Path (Get-Location) ".\.mamba\root"
    if (!(Test-Path $root)) { 
        New-Item -ItemType Directory -Path $root -Force | Out-Null 
    }
    $env:MAMBA_ROOT_PREFIX = $root

    if (Test-Path $EnvPath) {
        Write-Host "  ✓ Environment already exists: $EnvPath" -ForegroundColor Green
        $response = Read-Host "  既存の環境を再作成しますか？ (y/N)"
        if ($response -ne 'y' -and $response -ne 'Y') {
            Write-Host "  環境作成をスキップ" -ForegroundColor Yellow
            return
        }
        Write-Host "  既存環境を削除中..." -ForegroundColor Cyan
        Remove-Item -Recurse -Force $EnvPath
    }
    
    Write-Host "  Python 3.11環境を作成中..." -ForegroundColor Cyan
    & $MicromambaPath create -y -p $EnvPath -c conda-forge python=3.11 pip
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python環境の作成に失敗しました"
    }
    
    Write-Host "  ✓ Python環境作成完了" -ForegroundColor Green
}

function Install-Requirements {
    Write-Host "`n[3/3] パッケージのインストール..." -ForegroundColor Yellow
    
    # Upgrade pip
    Write-Host "  pipをアップグレード中..." -ForegroundColor Cyan
    & $MicromambaPath run -p $EnvPath python -m pip install --upgrade pip
    
    # Install core packages for simple_demo.py and check_qnn.py
    Write-Host "  必要なパッケージをインストール中..." -ForegroundColor Cyan
    
    $packages = @(
        "onnxruntime-qnn",  # QNN Execution Provider for NPU
        "opencv-python",     # Computer vision library
        "numpy",            # Numerical computing
        "python-dotenv"     # Environment variable management
    )
    
    foreach ($pkg in $packages) {
        Write-Host "    - $pkg" -ForegroundColor Gray
    }
    
    & $MicromambaPath run -p $EnvPath pip install $packages
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "パッケージのインストールに失敗しました"
    }
    
    Write-Host "  ✓ パッケージインストール完了" -ForegroundColor Green
}

function Verify-Installation {
    Write-Host "`n===== インストール確認 =====`n" -ForegroundColor Cyan
    
    Write-Host "Python バージョン:" -ForegroundColor Yellow
    & $MicromambaPath run -p $EnvPath python --version
    
    Write-Host "`nインストール済みパッケージ:" -ForegroundColor Yellow
    & $MicromambaPath run -p $EnvPath pip list | Select-String -Pattern "onnxruntime|opencv|numpy|dotenv"
    
    Write-Host "`nQNN Execution Provider 確認中..." -ForegroundColor Yellow
    $checkScript = @"
import onnxruntime as ort
providers = ort.get_available_providers()
if 'QNNExecutionProvider' in providers:
    print('  ✓ QNNExecutionProvider is available!')
else:
    print('  ✗ QNNExecutionProvider is NOT available')
print(f'  Available providers: {providers}')
"@
    
    & $MicromambaPath run -p $EnvPath python -c $checkScript
}

function Show-Usage {
    Write-Host "`n===== セットアップ完了 =====`n" -ForegroundColor Green
    
    Write-Host "次のコマンドで実行できます:`n" -ForegroundColor Yellow
    
    Write-Host "  1. QNN環境確認:" -ForegroundColor Cyan
    Write-Host "     .\.mamba\micromamba.exe run -p .\.venv-mamba python check_qnn.py`n" -ForegroundColor White
    
    Write-Host "  2. デモ実行:" -ForegroundColor Cyan
    Write-Host "     .\.mamba\micromamba.exe run -p .\.venv-mamba python simple_demo.py`n" -ForegroundColor White
    
    Write-Host "  3. SNPE変換（オプション）:" -ForegroundColor Cyan
    Write-Host "     .\scripts\debug_snpe.ps1`n" -ForegroundColor White
    
    Write-Host "または、通常のPythonで実行:" -ForegroundColor Yellow
    Write-Host "  python check_qnn.py" -ForegroundColor White
    Write-Host "  python simple_demo.py`n" -ForegroundColor White
}

# ===== Main Execution =====

try {
    Ensure-Micromamba
    Create-Env
    Install-Requirements
    Verify-Installation
    Show-Usage
    
    Write-Host "✓ すべてのセットアップが完了しました！`n" -ForegroundColor Green
    
} catch {
    Write-Host "`n❌ エラーが発生しました: $_`n" -ForegroundColor Red
    exit 1
}
