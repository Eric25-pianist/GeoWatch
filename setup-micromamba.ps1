param(
    [string]$RootPrefix = "",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not $RootPrefix) {
    $RootPrefix = Join-Path $ProjectRoot ".mamba-root"
}
$RootPrefix = [System.IO.Path]::GetFullPath($RootPrefix)

$MicromambaCommand = Get-Command micromamba -ErrorAction SilentlyContinue
if ($MicromambaCommand) {
    $Micromamba = $MicromambaCommand.Source
} else {
    $ToolsDirectory = Join-Path $ProjectRoot ".tools"
    $Archive = Join-Path $ToolsDirectory "micromamba.tar.bz2"
    $Micromamba = Join-Path $ToolsDirectory "Library\bin\micromamba.exe"
    if (-not (Test-Path $Micromamba)) {
        New-Item -ItemType Directory -Force -Path $ToolsDirectory | Out-Null
        Write-Host "Downloading a project-local Micromamba runtime..."
        Invoke-WebRequest `
            -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" `
            -OutFile $Archive
        tar -xjf $Archive -C $ToolsDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "Micromamba archive extraction failed with exit code $LASTEXITCODE."
        }
    }
    if (-not (Test-Path $Micromamba)) {
        throw "Micromamba executable was not found at $Micromamba."
    }
}

function Invoke-MicromambaChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Step,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host "`n[$Step]"
    & $script:Micromamba @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Step failed with exit code $exitCode. GeoWatch is not ready."
    }
}

$EnvironmentDirectory = Join-Path $RootPrefix "envs\geowatch"
if ($Recreate -and (Test-Path $EnvironmentDirectory)) {
    Invoke-MicromambaChecked "Removing existing environment" @(
        "--root-prefix", $RootPrefix, "env", "remove", "-n", "geowatch", "--yes"
    )
}

if (Test-Path (Join-Path $EnvironmentDirectory "python.exe")) {
    Invoke-MicromambaChecked "Updating Python 3.12 GIS environment" @(
        "--root-prefix", $RootPrefix, "env", "update", "-n", "geowatch",
        "--file", (Join-Path $ProjectRoot "environment.yml"), "--yes"
    )
} else {
    Invoke-MicromambaChecked "Creating Python 3.12 GIS environment" @(
        "--root-prefix", $RootPrefix, "create", "-n", "geowatch",
        "--file", (Join-Path $ProjectRoot "environment.yml"), "--yes"
    )
}

Invoke-MicromambaChecked "Installing GeoWatch into the GIS environment" @(
    "--root-prefix", $RootPrefix, "run", "-n", "geowatch", "python", "-m",
    "pip", "install", "-e", $ProjectRoot, "--no-deps"
)
Invoke-MicromambaChecked "Validating Python and GIS dependencies" @(
    "--root-prefix", $RootPrefix, "run", "-n", "geowatch", "geowatch",
    "doctor", "--strict"
)
Invoke-MicromambaChecked "Validating GeoWatch CLI" @(
    "--root-prefix", $RootPrefix, "run", "-n", "geowatch", "geowatch", "--help"
)

$LaunchCommand = "& '$Micromamba' --root-prefix '$RootPrefix' run -n geowatch geowatch wizard"
Write-Host "`nGeoWatch environment is ready."
Write-Host "Launch the application with:"
Write-Host $LaunchCommand
