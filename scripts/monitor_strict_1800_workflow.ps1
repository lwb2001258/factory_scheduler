[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$KeepScheduledTask
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$taskName = 'FactorySchedulerStrict1800Monitor'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$resultsDir = Join-Path $projectRoot `
    'results\all_scheduler_model_remediation\formal_1800_dual_v4'
$statePath = Join-Path $resultsDir 'workflow_state.json'
$monitorLog = Join-Path $resultsDir 'monitor_history.jsonl'
$monitorConsoleLog = Join-Path $resultsDir 'monitor_console.log'
$monitorStatePath = Join-Path $resultsDir 'monitor_state.json'
$runtimeProcessPath = Join-Path $resultsDir 'monitor_runtime_processes.json'
$workflowScript = Join-Path $projectRoot 'scripts\run_strict_1800_workflow.py'
$reportScript = Join-Path $projectRoot 'scripts\generate_strict_1800_report.py'
$reportPath = Join-Path $projectRoot `
    'docs\all_scheduler_model_remediation_training_report.md'
$python = 'C:\Python312\python.exe'
$staleProgressMinutes = 150.0
$expectedWorkflowContract = `
    'dual-objective-training-v10-unfloored-expert-bounded-supervision-strict'
$expectedStateSchema = 9

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $pythonCommand = Get-Command python.exe -ErrorAction Stop
    $python = $pythonCommand.Source
}

$createdNew = $false
$mutex = [Threading.Mutex]::new(
    $true,
    'Local\FactorySchedulerStrict1800Monitor',
    [ref]$createdNew)
if (-not $createdNew) {
    exit 0
}

$script:runtimeProcesses = @{
    workflow = $null
    report = $null
}
if (Test-Path -LiteralPath $runtimeProcessPath -PathType Leaf) {
    try {
        $runtimeDocument = Get-Content -LiteralPath $runtimeProcessPath -Raw |
            ConvertFrom-Json
        foreach ($role in @('workflow', 'report')) {
            if ($runtimeDocument.PSObject.Properties.Name -contains $role) {
                $script:runtimeProcesses[$role] = $runtimeDocument.$role
            }
        }
    }
    catch {
        $script:runtimeProcesses = @{ workflow = $null; report = $null }
    }
}

function Write-RuntimeProcesses {
    if ($DryRun) {
        return
    }
    $temporary = "$runtimeProcessPath.tmp"
    try {
        $script:runtimeProcesses | ConvertTo-Json -Depth 5 |
            Set-Content -LiteralPath $temporary -Encoding utf8
        Move-Item -LiteralPath $temporary -Destination $runtimeProcessPath -Force
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

function Set-TrackedProcess {
    param(
        [ValidateSet('workflow', 'report')][string]$Role,
        [int]$ProcessId
    )
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" `
        -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        $script:runtimeProcesses[$Role] = $null
    }
    else {
        $script:runtimeProcesses[$Role] = [ordered]@{
            process_id = [int]$process.ProcessId
            name = [string]$process.Name
            creation_date = ([datetime]$process.CreationDate).ToString('o')
        }
    }
    Write-RuntimeProcesses
}

function Get-TrackedProcess {
    param([ValidateSet('workflow', 'report')][string]$Role)
    $entry = $script:runtimeProcesses[$Role]
    if ($null -eq $entry) {
        return @()
    }
    $pidValue = [int](Get-JsonProperty $entry 'process_id' 0)
    if ($pidValue -le 0) {
        return @()
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" `
        -ErrorAction SilentlyContinue
    if ($null -eq $process -or $process.Name -ine 'python.exe') {
        return @()
    }
    $expectedCreation = [string](Get-JsonProperty $entry 'creation_date' '')
    $actualCreation = ([datetime]$process.CreationDate).ToString('o')
    if ($expectedCreation -and $actualCreation -ne $expectedCreation) {
        return @()
    }
    return @($process)
}

function Get-WorkflowProcesses {
    $escapedResults = [Regex]::Escape($resultsDir)
    $visible = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -ieq 'python.exe' -and
        $_.CommandLine -match '(?i)(^|[\\/])run_strict_1800_workflow\.py(?:\s|$)' -and
        $_.CommandLine -match $escapedResults
    })
    if ($visible.Count -gt 0) {
        return $visible
    }
    return @(Get-TrackedProcess 'workflow')
}

function Get-ReportProcesses {
    $escapedResults = [Regex]::Escape($resultsDir)
    $visible = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -ieq 'python.exe' -and
        $_.CommandLine -match '(?i)(^|[\\/])generate_strict_1800_report\.py(?:\s|$)' -and
        $_.CommandLine -match $escapedResults
    })
    if ($visible.Count -gt 0) {
        return $visible
    }
    return @(Get-TrackedProcess 'report')
}

function Get-JsonProperty {
    param($Document, [string]$Name, $Default = $null)
    if ($null -ne $Document -and
            $Document.PSObject.Properties.Name -contains $Name) {
        return $Document.$Name
    }
    return $Default
}

function Write-MonitorRecord {
    param([hashtable]$Record)
    $Record['recorded_at'] = (Get-Date).ToString('yyyy-MM-ddTHH:mm:sszzz')
    $json = $Record | ConvertTo-Json -Depth 8 -Compress
    if (-not $DryRun) {
        Add-Content -LiteralPath $monitorLog -Value $json -Encoding utf8
        $temporary = "$monitorStatePath.tmp"
        try {
            $Record | ConvertTo-Json -Depth 8 |
                Set-Content -LiteralPath $temporary -Encoding utf8
            Move-Item -LiteralPath $temporary -Destination $monitorStatePath -Force
        }
        finally {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        }
        # Start-Process holds its redirected stdout file exclusively on
        # Windows.  Keep a separate console transcript for scheduled runs;
        # interactive invocations still emit the same JSON below.
        Add-Content -LiteralPath $monitorConsoleLog `
            -Value "[HOURLY MONITOR] $json" -Encoding utf8
    }
    Write-Output $json
}

function Start-ReportWatcher {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stdout = Join-Path $resultsDir "report_generator.restart.$stamp.stdout.log"
    $stderr = Join-Path $resultsDir "report_generator.restart.$stamp.stderr.log"
    if ($DryRun) {
        return [ordered]@{
            started = $false
            dry_run = $true
            stdout = $stdout
            stderr = $stderr
        }
    }
    $process = Start-Process -FilePath $python -ArgumentList @(
        '-u', $reportScript,
        '--results-dir', $resultsDir,
        '--output', $reportPath,
        '--wait', '--poll-seconds', '60'
    ) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Set-TrackedProcess -Role report -ProcessId $process.Id
    return [ordered]@{
        started = $true
        process_id = $process.Id
        stdout = $stdout
        stderr = $stderr
    }
}

function Stop-WorkflowOrphans {
    $escapedResults = [Regex]::Escape($resultsDir)
    $orphans = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(?i:webots|webots-bin)\.exe$' -and
        $_.CommandLine -match $escapedResults
    })
    $stopped = @()
    foreach ($orphan in $orphans) {
        $stopped += $orphan.ProcessId
        if (-not $DryRun) {
            & taskkill.exe /PID $orphan.ProcessId /T /F *> $null
        }
    }
    return $stopped
}

function Start-WorkflowFromCheckpoint {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stdout = Join-Path $resultsDir "workflow.restart.$stamp.stdout.log"
    $stderr = Join-Path $resultsDir "workflow.restart.$stamp.stderr.log"
    if ($DryRun) {
        return [ordered]@{
            started = $false
            dry_run = $true
            stdout = $stdout
            stderr = $stderr
        }
    }
    $process = Start-Process -FilePath $python -ArgumentList @(
        '-u', $workflowScript,
        '--output-dir', $resultsDir
    ) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Set-TrackedProcess -Role workflow -ProcessId $process.Id
    return [ordered]@{
        started = $true
        process_id = $process.Id
        stdout = $stdout
        stderr = $stderr
    }
}

function Get-FailureDiagnosis {
    param($WorkflowState)
    $parts = @()
    $stateError = Get-JsonProperty $WorkflowState 'error'
    if ($stateError) {
        $parts += "state error: $stateError"
    }
    $stderrFiles = @(Get-ChildItem -LiteralPath $resultsDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'workflow*.stderr.log' -and $_.Length -gt 0 } |
        Sort-Object LastWriteTime -Descending)
    if ($stderrFiles.Count -gt 0) {
        $latest = $stderrFiles[0]
        $tail = @(Get-Content -LiteralPath $latest.FullName -Tail 20 -ErrorAction SilentlyContinue)
        if ($tail.Count -gt 0) {
            $parts += "latest stderr ($($latest.Name)): $($tail -join ' | ')"
        }
    }
    if ($parts.Count -eq 0) {
        return 'process exited without a committed error; treating as transient interruption'
    }
    return $parts -join '; '
}

function Get-CurrentTrainingResults {
    $headlessRoot = Join-Path $resultsDir 'headless'
    $policies = @()
    if (Test-Path -LiteralPath $headlessRoot -PathType Container) {
        foreach ($policyDir in @(Get-ChildItem -LiteralPath $headlessRoot `
                -Directory -ErrorAction SilentlyContinue | Sort-Object Name)) {
            if ($policyDir.Name -eq 'reference_baselines') {
                continue
            }
            $episodeRoot = Join-Path $policyDir.FullName 'episodes'
            $completedEpisodes = @(
                Get-ChildItem -LiteralPath $episodeRoot -Recurse `
                    -Filter result.json -File -ErrorAction SilentlyContinue
            ).Count
            $readiness = $null
            $readinessPath = Join-Path $policyDir.FullName `
                'standalone_readiness.json'
            if (Test-Path -LiteralPath $readinessPath -PathType Leaf) {
                try {
                    $readiness = Get-Content -LiteralPath $readinessPath -Raw |
                        ConvertFrom-Json
                }
                catch {
                    $readiness = $null
                }
            }
            $latestValidation = $null
            $validationRoot = Join-Path $policyDir.FullName 'validation'
            if (Test-Path -LiteralPath $validationRoot -PathType Container) {
                $roundDirs = @(Get-ChildItem -LiteralPath $validationRoot `
                    -Directory -ErrorAction SilentlyContinue | Where-Object {
                        $_.Name -match '^round_[0-9]+$'
                    } | Sort-Object {
                        [int]($_.Name -replace '^round_', '')
                    } -Descending)
                foreach ($roundDir in $roundDirs) {
                    $effectivePath = Join-Path $roundDir.FullName `
                        'effective_report.json'
                    $candidatePath = Join-Path $roundDir.FullName 'report.json'
                    $validationPath = if (
                            Test-Path -LiteralPath $effectivePath -PathType Leaf) {
                        $effectivePath
                    }
                    elseif (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
                        $candidatePath
                    }
                    else {
                        $null
                    }
                    if ($null -ne $validationPath) {
                        try {
                            $latestValidation = Get-Content `
                                -LiteralPath $validationPath -Raw |
                                ConvertFrom-Json
                            break
                        }
                        catch {
                            $latestValidation = $null
                        }
                    }
                }
            }
            $trainingReport = $null
            $trainingReportPath = Join-Path $policyDir.FullName `
                'training_report.json'
            if (Test-Path -LiteralPath $trainingReportPath -PathType Leaf) {
                try {
                    $trainingReport = Get-Content `
                        -LiteralPath $trainingReportPath -Raw |
                        ConvertFrom-Json
                }
                catch {
                    $trainingReport = $null
                }
            }
            $policies += [ordered]@{
                policy_id = $policyDir.Name
                completed_episodes = $completedEpisodes
                completed_balanced_rounds = if ($null -ne $readiness) {
                    Get-JsonProperty $readiness 'completed_balanced_rounds' 0
                }
                else {
                    [math]::Floor($completedEpisodes / 3)
                }
                readiness_status = if ($null -ne $readiness) {
                    Get-JsonProperty $readiness 'status' 'unknown'
                }
                else {
                    'not_yet_checked'
                }
                latest_validation_round = if ($null -ne $latestValidation) {
                    Get-JsonProperty $latestValidation 'round_index'
                }
                else {
                    $null
                }
                latest_validation_metric = if ($null -ne $latestValidation) {
                    Get-JsonProperty $latestValidation 'validation_metric'
                }
                else {
                    $null
                }
                promotion_accepted = if ($null -ne $latestValidation) {
                    Get-JsonProperty $latestValidation 'promotion_accepted'
                }
                else {
                    $null
                }
                standalone_complete = (
                    $null -ne $trainingReport -and
                    (Get-JsonProperty $trainingReport `
                        'webots_fine_tune_eligible' $false) -eq $true)
            }
        }
    }
    return [ordered]@{
        completed_standalone_models = @(
            $policies | Where-Object { $_.standalone_complete }
        ).Count
        total_ai_objective_models = 20
        policies = $policies
    }
}

try {
    if (-not (Test-Path -LiteralPath $resultsDir -PathType Container)) {
        throw "results directory does not exist: $resultsDir"
    }
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        throw "workflow state does not exist: $statePath"
    }

    $workflowState = Get-Content -LiteralPath $statePath -Raw |
        ConvertFrom-Json
    if ((Get-JsonProperty $workflowState 'workflow_contract') -ne
            $expectedWorkflowContract -or
            (Get-JsonProperty $workflowState 'schema_version') -ne
            $expectedStateSchema) {
        throw 'workflow state contract/schema differs from the fresh v4 run'
    }
    $status = [string](Get-JsonProperty $workflowState 'status' 'unknown')
    $stage = [string](Get-JsonProperty $workflowState 'stage' 'unknown')
    $current = Get-JsonProperty $workflowState 'current'
    $evaluationCount = @(
        Get-ChildItem -LiteralPath (Join-Path $resultsDir 'evaluation\episodes') `
            -Recurse -Filter result.json -File -ErrorAction SilentlyContinue
    ).Count
    $mainProcesses = @(Get-WorkflowProcesses)
    $reportProcesses = @(Get-ReportProcesses)
    if ($mainProcesses.Count -gt 0) {
        $mainProcesses = @($mainProcesses | Sort-Object CreationDate)
        Set-TrackedProcess -Role workflow -ProcessId $mainProcesses[0].ProcessId
    }
    if ($reportProcesses.Count -gt 0) {
        $reportProcesses = @($reportProcesses | Sort-Object CreationDate)
        Set-TrackedProcess -Role report -ProcessId $reportProcesses[0].ProcessId
    }

    $removedDuplicateWorkflows = @()
    if ($mainProcesses.Count -gt 1) {
        foreach ($duplicate in @($mainProcesses | Select-Object -Skip 1)) {
            $removedDuplicateWorkflows += $duplicate.ProcessId
            if (-not $DryRun) {
                & taskkill.exe /PID $duplicate.ProcessId /T /F *> $null
            }
        }
        $mainProcesses = @($mainProcesses | Select-Object -First 1)
    }
    $removedDuplicateReports = @()
    if ($reportProcesses.Count -gt 1) {
        foreach ($duplicate in @($reportProcesses | Select-Object -Skip 1)) {
            $removedDuplicateReports += $duplicate.ProcessId
            if (-not $DryRun) {
                Stop-Process -Id $duplicate.ProcessId -Force -ErrorAction SilentlyContinue
            }
        }
        $reportProcesses = @($reportProcesses | Select-Object -First 1)
    }

    $record = [ordered]@{
        status = $status
        stage = $stage
        current = $current
        process_check = [ordered]@{
            workflow_running = ($mainProcesses.Count -gt 0)
            report_watcher_running = ($reportProcesses.Count -gt 0)
            checked_at = (Get-Date).ToString('yyyy-MM-ddTHH:mm:sszzz')
        }
        current_results = Get-CurrentTrainingResults
        evaluation_completed = $evaluationCount
        evaluation_total = 870
        workflow_process_ids = @($mainProcesses | ForEach-Object ProcessId)
        report_process_ids = @($reportProcesses | ForEach-Object ProcessId)
        removed_duplicate_workflow_ids = $removedDuplicateWorkflows
        removed_duplicate_report_ids = $removedDuplicateReports
        action = 'none'
    }

    if ($status -eq 'complete' -and $stage -eq 'complete') {
        if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
            $record['action'] = 'complete_report_present'
            $record['report'] = $reportPath
            if (-not $KeepScheduledTask -and -not $DryRun) {
                & schtasks.exe /Change /TN $taskName /DISABLE *> $null
                $record['scheduled_task_disabled'] = $true
            }
        }
        elseif ($reportProcesses.Count -gt 0) {
            $record['action'] = 'complete_waiting_for_report_generator'
        }
        else {
            $record['action'] = 'complete_started_report_generator'
            $record['report_watcher'] = Start-ReportWatcher
        }
        Write-MonitorRecord $record
        exit 0
    }

    if ($mainProcesses.Count -gt 0) {
        $stateAgeMinutes = (
            (Get-Date) - (Get-Item -LiteralPath $statePath).LastWriteTime
        ).TotalMinutes
        $record['state_age_minutes'] = [math]::Round($stateAgeMinutes, 2)
        if ($stateAgeMinutes -gt $staleProgressMinutes) {
            $record['action'] = 'stale_process_restart_from_checkpoint'
            $record['diagnosis'] = (
                "workflow state made no progress for " +
                "$([math]::Round($stateAgeMinutes, 2)) minutes")
            $record['stopped_stale_workflow_ids'] = @(
                $mainProcesses | ForEach-Object {
                    if (-not $DryRun) {
                        & taskkill.exe /PID $_.ProcessId /T /F *> $null
                    }
                    $_.ProcessId
                })
            $record['stopped_orphan_process_ids'] = @(Stop-WorkflowOrphans)
            $record['restart'] = Start-WorkflowFromCheckpoint
            if ($reportProcesses.Count -eq 0) {
                $record['report_watcher'] = Start-ReportWatcher
            }
            Write-MonitorRecord $record
            exit 0
        }
        $record['action'] = 'healthy'
        if ($reportProcesses.Count -eq 0) {
            $record['action'] = 'healthy_restarted_report_watcher'
            $record['report_watcher'] = Start-ReportWatcher
        }
        Write-MonitorRecord $record
        exit 0
    }

    $record['action'] = 'restart_from_checkpoint'
    $record['diagnosis'] = Get-FailureDiagnosis $workflowState
    $record['stopped_orphan_process_ids'] = @(Stop-WorkflowOrphans)
    $record['restart'] = Start-WorkflowFromCheckpoint
    if ($reportProcesses.Count -eq 0) {
        $record['report_watcher'] = Start-ReportWatcher
    }
    Write-MonitorRecord $record
}
catch {
    $failureRecord = [ordered]@{
        status = 'monitor_error'
        action = 'none'
        error = $_.Exception.Message
        stack = $_.ScriptStackTrace
    }
    Write-MonitorRecord $failureRecord
    exit 1
}
finally {
    if ($null -ne $mutex) {
        $mutex.ReleaseMutex() | Out-Null
        $mutex.Dispose()
    }
}
