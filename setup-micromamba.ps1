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

function Get-TarExecutable {
    $TarCommand = Get-Command tar.exe -ErrorAction SilentlyContinue
    if (-not $TarCommand) {
        $TarCommand = Get-Command tar -ErrorAction SilentlyContinue
    }
    if (-not $TarCommand) {
        throw (
            "Windows tar.exe was not found. Install a recent version of Windows " +
            "or extract the project on a system where tar.exe is available."
        )
    }
    return $TarCommand.Source
}

function Install-ProjectMicromamba {
    param(
        [Parameter(Mandatory = $true)][string]$ToolsDirectory,
        [Parameter(Mandatory = $true)][string]$Archive,
        [Parameter(Mandatory = $true)][string]$Micromamba
    )

    New-Item -ItemType Directory -Force -Path $ToolsDirectory | Out-Null
    $TarExecutable = Get-TarExecutable
    $DownloadUrl = "https://micro.mamba.pm/api/micromamba/win-64/latest"

    foreach ($Attempt in 1..2) {
        if ((-not (Test-Path $Archive)) -or $Attempt -gt 1) {
            if (Test-Path $Archive) {
                Remove-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue
            }
            Write-Host "Downloading a project-local Micromamba runtime..."
            Invoke-WebRequest -Uri $DownloadUrl -OutFile $Archive
        }

        foreach ($PartialPath in @(
            (Join-Path $ToolsDirectory "Library"),
            (Join-Path $ToolsDirectory "info")
        )) {
            if (Test-Path $PartialPath) {
                Remove-Item -LiteralPath $PartialPath -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        & $TarExecutable -xjf $Archive -C $ToolsDirectory
        $ExtractExitCode = $LASTEXITCODE
        if ($ExtractExitCode -eq 0 -and (Test-Path $Micromamba)) {
            return
        }
    }

    throw (
        "Micromamba archive extraction failed. Delete the .tools folder and try " +
        "again. If the issue continues, run tar.exe manually to inspect the " +
        "Windows extraction error."
    )
}

$MicromambaCommand = Get-Command micromamba -ErrorAction SilentlyContinue
if ($MicromambaCommand) {
    $Micromamba = $MicromambaCommand.Source
} else {
    $ToolsDirectory = Join-Path $ProjectRoot ".tools"
    $Archive = Join-Path $ToolsDirectory "micromamba.tar.bz2"
    $Micromamba = Join-Path $ToolsDirectory "Library\bin\micromamba.exe"
    if (-not (Test-Path $Micromamba)) {
        Install-ProjectMicromamba `
            -ToolsDirectory $ToolsDirectory `
            -Archive $Archive `
            -Micromamba $Micromamba
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
