param([switch]$SkipPetscBuild)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $taskRoot
$taskVenv = Join-Path $taskRoot '.venv'
$taskPython = Join-Path $taskVenv 'Scripts\python.exe'
$taskBuild = Join-Path $taskVenv 'build'
$taskTools = Join-Path $taskVenv 'build-tools'
$taskCygwin = Join-Path $taskTools 'cygwin'
$taskSource = Join-Path $taskBuild 'petsc-3.26.1'
$taskPrefix = Join-Path $taskVenv 'petsc'

function Assert-TaskExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)." }
}

if (-not (Test-Path -LiteralPath $taskPython)) {
    & python -m venv $taskVenv
    Assert-TaskExit 'Creating the virtual environment'
}
New-Item -ItemType Directory -Force -Path $taskBuild, $taskTools, "$taskVenv\tmp" | Out-Null
$env:TEMP = "$taskVenv\tmp"
$env:TMP = $env:TEMP
& $taskPython -m pip install --disable-pip-version-check --no-cache-dir numpy==2.5.4 torch==2.15.0 setuptools==84.2.1 wheel==0.48.0 cython==3.3.1
Assert-TaskExit 'Installing Python dependencies'

if (-not $SkipPetscBuild) {
    $taskVswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $taskVswhere)) { throw 'Visual Studio C++ Build Tools are required.' }
    $taskVisualStudio = & $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $taskVisualStudio) { throw 'Visual Studio C++ Build Tools were not found.' }
    $taskCompilerSetup = Join-Path $taskVisualStudio 'Common7\Tools\VsDevCmd.bat'
    $taskCompilerCommand = '"{0}" -arch=x64 -host_arch=x64 >nul && set' -f $taskCompilerSetup
    $taskCompilerEnvironment = & cmd.exe /d /s /c $taskCompilerCommand
    Assert-TaskExit 'Initialising MSVC'
    foreach ($taskLine in $taskCompilerEnvironment) {
        if ($taskLine -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }

    $taskInstaller = Join-Path $taskTools 'setup-x86_64.exe'
    if (-not (Test-Path -LiteralPath $taskInstaller)) {
        Invoke-WebRequest 'https://cygwin.com/setup-x86_64.exe' -OutFile $taskInstaller
    }
    $taskInstallerArgs = @(
        '--quiet-mode', 'hidden', '--no-admin', '--no-shortcuts',
        '--no-write-registry', '--no-replaceonreboot', '--only-site',
        '--site', 'https://mirrors.dotsrc.org/cygwin/',
        '--root', ('"{0}"' -f $taskCygwin),
        '--local-package-dir', ('"{0}\cache"' -f $taskTools),
        '--packages', 'python3,make,diffutils,patch'
    )
    $taskSetupProcess = Start-Process -FilePath $taskInstaller -ArgumentList $taskInstallerArgs -WindowStyle Hidden -Wait -PassThru
    if ($taskSetupProcess.ExitCode -ne 0) { throw "Cygwin setup failed (exit $($taskSetupProcess.ExitCode))." }

    if (-not (Test-Path -LiteralPath "$taskSource\configure")) {
        $taskDownloadCode = @'
import hashlib, json, pathlib, sys, tarfile, urllib.request
target = pathlib.Path(sys.argv[1])
meta = json.load(urllib.request.urlopen('https://pypi.org/pypi/petsc/3.26.1/json'))
item = next(x for x in meta['urls'] if x['packagetype'] == 'sdist')
archive = target / item['filename']
urllib.request.urlretrieve(item['url'], archive)
if hashlib.sha256(archive.read_bytes()).hexdigest() != item['digests']['sha256']:
    raise RuntimeError('PETSc source checksum mismatch')
with tarfile.open(archive) as source:
    source.extractall(target, filter='data')
'@
        & $taskPython -c $taskDownloadCode $taskBuild
        Assert-TaskExit 'Downloading PETSc'
    }

    $taskShellCode = @'
#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/bin:$PATH"
cd "$(cygpath -u "$1")"
export PETSC_DIR="$PWD"
export PETSC_ARCH=arch-mswin-c-opt
/usr/bin/python3 ./configure \
    --with-cc=cl --with-cxx=0 --with-fc=0 \
    --with-mpi=0 --with-debugging=0 --with-shared-libraries=0 \
    'CFLAGS=-MD -GF -wd4996 -Zc:preprocessor' 'COPTFLAGS=-O2' \
    --with-scalar-type=real --with-precision=double \
    --download-f2cblaslapack --with-make-np=8
/usr/bin/make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
/usr/bin/make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" check
'@
    $taskShellPath = Join-Path $taskBuild 'build_petsc.sh'
    [IO.File]::WriteAllText($taskShellPath, $taskShellCode.Replace("`r`n", "`n") + "`n", [Text.UTF8Encoding]::new($false))
    $taskPosixShellPath = & "$taskCygwin\bin\cygpath.exe" -u $taskShellPath
    Assert-TaskExit 'Converting the build script path'
    & "$taskCygwin\bin\bash.exe" --noprofile --norc $taskPosixShellPath $taskSource *> "$taskVenv\build-native-petsc.log"
    Assert-TaskExit 'Building and checking PETSc (see .venv/build-native-petsc.log)'
}

& $taskPython "$PSScriptRoot\stage_petsc_windows.py" $taskSource arch-mswin-c-opt $taskPrefix
Assert-TaskExit 'Staging PETSc for Windows Python'
$env:PETSC_DIR = $taskPrefix
[Environment]::SetEnvironmentVariable('PETSC_ARCH', $null, 'Process')
$env:CL = "$env:CL /Zc:preprocessor"
& $taskPython -m pip install --disable-pip-version-check --no-cache-dir --no-build-isolation --no-deps "$taskSource\src\binding\petsc4py" --log "$taskVenv\build-petsc4py.log" *> "$taskVenv\build-petsc4py-console.log"
Assert-TaskExit 'Building petsc4py (see .venv/build-petsc4py.log)'
& $taskPython -m pip install --no-cache-dir --no-build-isolation --no-deps -e $taskRoot
Assert-TaskExit 'Installing the project'
& $taskPython "$PSScriptRoot\check_environment.py"
Assert-TaskExit 'Checking the environment'
& $taskPython -m pip check
Assert-TaskExit 'Checking Python dependencies'
