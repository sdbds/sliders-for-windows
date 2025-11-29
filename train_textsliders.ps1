# Sliders train script by @bdsqlsz

#SD version(Lora、Sdxl_lora、hunyuan_lora、zimage_lora)
$sd_version = "zimage_lora"

#train_mode(text,image)
$train_mode = "text"

$compile = $true

#output name
$name = "zimage_test"

# Train data config | 设置训练配置路径
$config_file = "./trainscripts/textsliders/data/config_zimage.yaml" # config path | 配置路径

# main body attributes| 主体属性
$attributes = ''

#LoRA rank and alpha
$rank = 64
$alpha = 1

#image model
$folder_main = 'datasets/eyesize/'
$folders = "bigsize, smallsize"
$scales = "1, -1"

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
  if ($compile) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = & $vswhere -latest -products * `
      -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
      -property installationPath
    & (Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1") -Arch amd64
    Set-Location $PSScriptRoot
  }
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:VSLANG = "1033"
$Env:NVIDIA_TF32_OVERRIDE="1"  # 启用 TF32 加速（Ampere+ GPU）
$ext_args = [System.Collections.ArrayList]::new()

if ($train_mode -ieq "text") {
  $laungh_script = "textsliders/train_lora"
}
else {
  $laungh_script = "imagesliders/train_lora-scale"
  [void]$ext_args.Add("--folder_main=$folder_main")
  [void]$ext_args.Add("--folders=$folders")
  [void]$ext_args.Add("--scales=$scales")
}

if ($sd_version -ilike "sdxl*") {
  $laungh_script = $laungh_script + "_xl"
}

if ($sd_version -ilike "hunyuan*") {
  $laungh_script = $laungh_script + "_hunyuan"
}

if ($sd_version -ilike "zimage*") {
  $laungh_script = $laungh_script + "_zimage"
}

# run train
python -m accelerate.commands.launch --num_cpu_threads_per_process=8 "./trainscripts/$laungh_script.py" `
  --config_file=$config_file `
  --attributes=$attributes `
  --name=$name `
  --rank=$rank `
  --alpha=$alpha $ext_args

Write-Output "Train finished"
Read-Host | Out-Null ;