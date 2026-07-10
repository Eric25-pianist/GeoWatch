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

function Add-DirectoryToUserPath {
    param(
        [Parameter(Mandatory = $true)][string]$Directory
    )

    $ResolvedDirectory = [System.IO.Path]::GetFullPath($Directory)
    $CurrentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $ExistingSegments = @()
    if ($CurrentUserPath) {
        $ExistingSegments = $CurrentUserPath -split ";" |
            Where-Object { $_ -and $_.Trim() }
    }
    $FilteredUserSegments = $ExistingSegments |
        Where-Object { $_.TrimEnd("\") -ine $ResolvedDirectory.TrimEnd("\") }
    $NewUserSegments = @($ResolvedDirectory) + $FilteredUserSegments
    $NewUserPath = ($NewUserSegments | Where-Object { $_ }) -join ";"
    if ($NewUserPath -ne $CurrentUserPath) {
        [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
    }

    $CurrentProcessPath = [Environment]::GetEnvironmentVariable("Path", "Process")
    $ProcessSegments = @()
    if ($CurrentProcessPath) {
        $ProcessSegments = $CurrentProcessPath -split ";" |
            Where-Object { $_ -and $_.Trim() }
    }
    $FilteredProcessSegments = $ProcessSegments |
        Where-Object { $_.TrimEnd("\") -ine $ResolvedDirectory.TrimEnd("\") }
    $env:Path = (@($ResolvedDirectory) + $FilteredProcessSegments) -join ";"
}

function Install-GeowatchLauncher {
    param(
        [Parameter(Mandatory = $true)][string]$RootPrefix
    )

    $EnvironmentPrefix = Join-Path $RootPrefix "envs\geowatch"
    $GeowatchExecutable = Join-Path $EnvironmentPrefix "Scripts\geowatch.exe"
    if (-not (Test-Path $GeowatchExecutable)) {
        throw "GeoWatch executable was not found at $GeowatchExecutable."
    }

    $LauncherDirectory = Join-Path $env:LOCALAPPDATA "GeoWatch\bin"
    New-Item -ItemType Directory -Force -Path $LauncherDirectory | Out-Null
    $LauncherPath = Join-Path $LauncherDirectory "geowatch.cmd"
    $ObsoleteLauncherExecutable = Join-Path $LauncherDirectory "geowatch.exe"
    Remove-Item -LiteralPath $ObsoleteLauncherExecutable -Force -ErrorAction SilentlyContinue
    $LauncherContent = @"
@echo off
set "GEOWATCH_ENV=$EnvironmentPrefix"
if not exist "%GEOWATCH_ENV%\Scripts\geowatch.exe" (
  echo GeoWatch environment was not found at "%GEOWATCH_ENV%".
  echo Run setup-micromamba.ps1 again from the GeoWatch project folder.
  exit /b 1
)
set "PATH=%GEOWATCH_ENV%;%GEOWATCH_ENV%\Library\bin;%GEOWATCH_ENV%\Scripts;%PATH%"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
if exist "%GEOWATCH_ENV%\Library\share\gdal" set "GDAL_DATA=%GEOWATCH_ENV%\Library\share\gdal"
if exist "%GEOWATCH_ENV%\Library\share\proj" set "PROJ_LIB=%GEOWATCH_ENV%\Library\share\proj"
"%GEOWATCH_ENV%\Scripts\geowatch.exe" %*
exit /b %ERRORLEVEL%
"@
    Set-Content -Path $LauncherPath -Value $LauncherContent -Encoding ASCII
    Add-DirectoryToUserPath -Directory $LauncherDirectory
    Install-PowerShellShortcut -LauncherPath $LauncherPath
    return $LauncherPath
}

function Install-PowerShellShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$LauncherPath
    )

    $EscapedLauncher = $LauncherPath.Replace("'", "''")
    $CurrentFunction = [scriptblock]::Create("& '$EscapedLauncher' @args")
    Set-Item -Path Function:\global:geowatch -Value $CurrentFunction -Force

    $ProfilePath = $PROFILE.CurrentUserAllHosts
    if (-not $ProfilePath) {
        return
    }
    $ProfileDirectory = Split-Path -Parent $ProfilePath
    New-Item -ItemType Directory -Force -Path $ProfileDirectory | Out-Null

    $Block = @"
# >>> GeoWatch launcher >>>
function geowatch {
    & '$EscapedLauncher' @args
}
# <<< GeoWatch launcher <<<
"@
    $Existing = ""
    if (Test-Path $ProfilePath) {
        $Existing = Get-Content -LiteralPath $ProfilePath -Raw
    }
    $Pattern = "(?s)# >>> GeoWatch launcher >>>.*?# <<< GeoWatch launcher <<<"
    if ($Existing -match $Pattern) {
        $Updated = [regex]::Replace($Existing, $Pattern, $Block)
    } elseif ($Existing.Trim()) {
        $Updated = $Existing.TrimEnd() + [Environment]::NewLine +
            [Environment]::NewLine + $Block + [Environment]::NewLine
    } else {
        $Updated = $Block + [Environment]::NewLine
    }
    Set-Content -LiteralPath $ProfilePath -Value $Updated -Encoding UTF8
}

function Stop-StaleGeoWatchProcesses {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$RootPrefix
    )

    $EnvironmentPrefix = Join-Path $RootPrefix "envs\geowatch"
    $ToolsPrefix = Join-Path $ProjectRoot ".tools"
    $Prefixes = @(
        [System.IO.Path]::GetFullPath($EnvironmentPrefix).ToLowerInvariant(),
        [System.IO.Path]::GetFullPath($ToolsPrefix).ToLowerInvariant()
    )

    $Processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            if (-not $_.ExecutablePath -or -not $_.CommandLine) {
                return $false
            }
            $ExecutablePath = $_.ExecutablePath.ToLowerInvariant()
            $InsideProjectRuntime = $false
            foreach ($Prefix in $Prefixes) {
                if ($ExecutablePath.StartsWith($Prefix)) {
                    $InsideProjectRuntime = $true
                    break
                }
            }
            return $InsideProjectRuntime -and
                $_.CommandLine.ToLowerInvariant().Contains("geowatch")
        }

    foreach ($Process in $Processes) {
        if ($Process.ProcessId -eq $PID) {
            continue
        }
        Write-Host (
            "Stopping stale GeoWatch process $($Process.ProcessId) " +
            "before reinstalling the command launcher..."
        )
        Stop-Process -Id $Process.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Remove-StalePipArtifacts {
    param(
        [Parameter(Mandatory = $true)][string]$RootPrefix
    )

    $SitePackages = Join-Path $RootPrefix "envs\geowatch\Lib\site-packages"
    if (-not (Test-Path $SitePackages)) {
        return
    }
    Get-ChildItem -LiteralPath $SitePackages -Force -Filter "~eowatch*" |
        ForEach-Object {
            Write-Host "Removing stale interrupted pip artifact: $($_.Name)"
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
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

Stop-StaleGeoWatchProcesses -ProjectRoot $ProjectRoot -RootPrefix $RootPrefix
Remove-StalePipArtifacts -RootPrefix $RootPrefix
Invoke-MicromambaChecked "Installing GeoWatch into the GIS environment" @(
    "--root-prefix", $RootPrefix, "run", "-n", "geowatch", "python", "-m",
    "pip", "install", "--upgrade", "--force-reinstall", "-e", $ProjectRoot,
    "--no-deps"
)
Invoke-MicromambaChecked "Validating Python and GIS dependencies" @(
    "--root-prefix", $RootPrefix, "run", "-n", "geowatch", "geowatch",
    "doctor", "--strict"
)
Invoke-MicromambaChecked "Validating GeoWatch CLI" @(
    "--root-prefix", $RootPrefix, "run", "-n", "geowatch", "geowatch", "--help"
)

$LauncherPath = Install-GeowatchLauncher -RootPrefix $RootPrefix
$LaunchCommand = "& '$Micromamba' --root-prefix '$RootPrefix' run -n geowatch geowatch wizard"
Write-Host "`nGeoWatch environment is ready."
Write-Host "Installed command shortcut:"
Write-Host "geowatch"
Write-Host "Shortcut location: $LauncherPath"
Write-Host "`nLaunch the application with:"
Write-Host "geowatch"
Write-Host "`nThe direct Micromamba command still works:"
Write-Host $LaunchCommand
