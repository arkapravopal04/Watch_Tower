param([switch]$FetchCache, [switch]$IndependentCheck)

$ErrorActionPreference = 'Stop'
$verificationRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $verificationRoot
$localLeanBin = Join-Path $workspaceRoot 'outputs\lean-tools\lean-4.35.0-windows\bin'
if (Test-Path (Join-Path $localLeanBin 'lake.exe')) {
    $env:PATH = $localLeanBin + [IO.Path]::PathSeparator + $env:PATH
}
if (-not (Get-Command lake -ErrorAction SilentlyContinue)) {
    throw 'Install the toolchain in lean-toolchain, or unpack the official Lean 4.35.0 Windows release under outputs/lean-tools.'
}
$env:MATHLIB_CACHE_DIR = Join-Path $workspaceRoot 'outputs\lean-tools\mathlib-cache'
$env:MATHLIB_NO_CACHE_ON_UPDATE = '1'

Push-Location $verificationRoot
try {
    # Only authored source files; dependency libraries have their own audits.
    $sources = @(Get-ChildItem -LiteralPath (Join-Path $verificationRoot 'ProtectedDoRA') -Filter '*.lean')
    $sources += Get-Item (Join-Path $verificationRoot 'ProtectedDoRA.lean')
    $holes = $sources | Select-String -Pattern '\b(sorry|admit|axiom)\b'
    if ($holes) { throw "Unfinished proof or custom axiom in authored sources: $holes" }

    # Native tools write progress to stderr; judge them by exit code only.
    $ErrorActionPreference = 'Continue'
    if ($FetchCache) {
        if (-not (Test-Path 'lake-manifest.json')) {
            & lake update
            if ($LASTEXITCODE -ne 0) { throw 'Dependency resolution failed.' }
        }
        # Fetch prebuilt Mathlib files for exactly the modules the proofs import.
        $mathlibImports = @($sources | Select-String -Pattern '^import (Mathlib\.\S+)' |
            ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique)
        & lake exe cache get @mathlibImports
        if ($LASTEXITCODE -ne 0) { throw 'Mathlib cache fetch failed.' }
    }
    & lake build
    if ($LASTEXITCODE -ne 0) { throw 'Lean build failed.' }

    # AxiomAudit.lean enumerates every theorem in the ProtectedDoRA modules and
    # reports an error for any dependency beyond propext, Classical.choice and
    # Quot.sound (sorryAx included), so a nonzero exit means a failed audit.
    $audit = @(& lake env lean AxiomAudit.lean 2>&1 | ForEach-Object { "$_" })
    $auditExit = $LASTEXITCODE
    $audit | ForEach-Object { Write-Output $_ }
    if ($auditExit -ne 0) { throw 'Axiom audit failed.' }
    $text = $audit -join "`n"
    if ($text -match '\bsorryAx\b') { throw 'Proof dependency contains sorryAx.' }
    if ($text -notmatch 'audited \d+ theorems: all use only propext, Classical.choice, Quot.sound') {
        throw 'Axiom audit did not report a clean result.'
    }
    Write-Output 'Verified: library compiled; every ProtectedDoRA theorem uses only the standard axioms.'

    if ($IndependentCheck) {
        # Replay every compiled module through the kernel a second time. One at a time on
        # purpose: each process loads all of Mathlib, and parallel runs measured slower.
        $modules = @('ProtectedDoRA') + @(Get-ChildItem -LiteralPath (Join-Path $verificationRoot 'ProtectedDoRA') -Filter '*.lean' |
            ForEach-Object { 'ProtectedDoRA.' + $_.BaseName })
        foreach ($module in $modules) {
            & lake env leanchecker $module
            if ($LASTEXITCODE -ne 0) { throw "leanchecker rejected $module." }
        }
        Write-Output "leanchecker: replayed $($modules.Count) modules."

        # Export each audited theorem with its whole dependency closure and check
        # it with nanoda, an independent kernel, allowing only the standard axioms.
        $theorems = @($audit | Select-String -Pattern '^(ProtectedDoRA\S*) depends on axioms' |
            ForEach-Object { $_.Matches[0].Groups[1].Value })
        $checkDir = Join-Path $workspaceRoot 'outputs\lean-tools\independent-check'
        New-Item -ItemType Directory -Force $checkDir | Out-Null
        $exportPath = Join-Path $checkDir 'export.ndjson'
        # cmd keeps the exporter's bytes; PowerShell 5.1 redirection would re-encode them.
        & cmd /c "lake env leanexport ProtectedDoRA -- $($theorems -join ' ') > `"$exportPath`""
        if ($LASTEXITCODE -ne 0) { throw 'leanexport failed.' }
        $config = @{
            export_file_path = $exportPath.Replace('\', '/')
            use_stdin = $false
            permitted_axioms = @('propext', 'Classical.choice', 'Quot.sound')
            unpermitted_axiom_hard_error = $true
            nat_extension = $true
            string_extension = $true
            print_success_message = $true
        } | ConvertTo-Json
        $configPath = Join-Path $checkDir 'nanoda.json'
        [IO.File]::WriteAllText($configPath, $config, (New-Object Text.UTF8Encoding $false))
        $nanoda = @(& (Join-Path $localLeanBin 'nanoda_bin.exe') $configPath 2>&1 | ForEach-Object { "$_" })
        $nanodaExit = $LASTEXITCODE
        $nanoda | ForEach-Object { Write-Output $_ }
        if ($nanodaExit -ne 0 -or -not (($nanoda -join "`n") -match 'with no typechecker errors')) {
            throw 'nanoda rejected the export.'
        }
        Remove-Item -LiteralPath $exportPath
        Write-Output "nanoda: independently checked the closure of $($theorems.Count) theorems."
    }
}
finally {
    Pop-Location
}
