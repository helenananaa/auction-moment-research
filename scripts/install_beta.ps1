param(
    [string]$PackageDirectory = $PSScriptRoot,
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
$packageRoot = (Resolve-Path -LiteralPath $PackageDirectory).Path
$manifestPath = Join-Path $packageRoot "beta-release-manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($record in $manifest.files) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $packageRoot $record.path))
    if (-not $candidate.StartsWith($packageRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest path outside the package: $($record.path)"
    }
    $file = Get-Item -LiteralPath $candidate
    if ($file.Length -ne $record.bytes -or (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) {
        throw "Package checksum mismatch: $($record.path)"
    }
}
$environment = Join-Path $packageRoot ".venv"
if (Test-Path -LiteralPath $environment) {
    throw "The package already contains .venv. Extract to a new directory to install safely."
}
if ($Python) { & $Python -m venv $environment }
else { & py -3.12 -m venv $environment }
if ($LASTEXITCODE -ne 0) { throw "Could not create the Python environment." }
$runtime = Join-Path $environment "Scripts\python.exe"
& $runtime -m pip install (Join-Path $packageRoot "wheels\auction_moment_research-0.1.1-py3-none-any.whl") (Join-Path $packageRoot "wheels\auction_moment_assistant-0.2.0b7-py3-none-any.whl")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $runtime -m pip check
if ($LASTEXITCODE -ne 0) { throw "Installed dependencies are inconsistent." }
$assistant = Join-Path $environment "Scripts\auction-vision-assistant.exe"
& $assistant --models (Join-Path $packageRoot "models\latest-model-v6-v2.0.0-beta.7") --verify-models
if ($LASTEXITCODE -ne 0) { throw "Model verification failed." }
Write-Host "Installation complete. Run start-assistant.cmd when ready."
