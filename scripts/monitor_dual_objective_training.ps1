param(
    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"

$workflowScript = Join-Path $RepositoryRoot "scripts\run_strict_1800_workflow.py"
$outputDir = Join-Path $RepositoryRoot "results\all_scheduler_model_remediation\formal_1800_dual_v4"
$statePath = Join-Path $outputDir "workflow_state.json"
$reportPath = Join-Path $RepositoryRoot "docs\all_scheduler_model_remediation_training_report.md"
$webotsPath = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
$monitorDir = Join-Path $outputDir "monitor"
$historyPath = Join-Path $monitorDir "status_history.jsonl"
$pidRecordPath = Join-Path $monitorDir "workflow_pid.json"
$pythonPath = "C:\Python312\python.exe"
$staleActivityMinutes = 90

New-Item -ItemType Directory -Path $monitorDir -Force | Out-Null

function Write-MonitorRecord {
    param([hashtable]$Record)

    $Record.checked_at = (Get-Date).ToString("o")
    Add-Content -LiteralPath $historyPath -Value ($Record | ConvertTo-Json -Compress -Depth 8) -Encoding UTF8
}

function Save-WorkflowPid {
    param([int]$ProcessId)

    @{
        process_id = $ProcessId
        recorded_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $pidRecordPath -Encoding UTF8
}

$mutex = [Threading.Mutex]::new($false, "Local\factory_scheduler_dual_v4_monitor")
$lockTaken = $false

try {
    $lockTaken = $mutex.WaitOne(0)
    if (-not $lockTaken) {
        Write-MonitorRecord @{ status = "skipped_overlapping_monitor" }
        exit 0
    }

    if (-not (Test-Path -LiteralPath $statePath)) {
        Write-MonitorRecord @{
            status = "state_missing"
            state_path = $statePath
        }
        exit 2
    }

    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.workflow_contract -ne "dual-objective-training-v10-unfloored-expert-bounded-supervision-strict" -or
            $state.schema_version -ne 9) {
        Write-MonitorRecord @{
            status = "unexpected_workflow_contract"
            workflow_contract = $state.workflow_contract
            state_status = $state.status
        }
        exit 2
    }

    $knownProcess = $null
    if (Test-Path -LiteralPath $pidRecordPath) {
        try {
            $knownPid = (Get-Content -LiteralPath $pidRecordPath -Raw | ConvertFrom-Json).process_id
            if ($null -ne $knownPid) {
                $candidate = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$knownPid) -ErrorAction SilentlyContinue
                if ($null -ne $candidate -and
                    $candidate.Name -eq "python.exe" -and
                    $candidate.CommandLine -like "*run_strict_1800_workflow.py*" -and
                    $candidate.CommandLine -like "*formal_1800_dual_v4*") {
                    $knownProcess = $candidate
                }
            }
        }
        catch {
            $knownProcess = $null
        }
    }

    $workflowProcessIds = @()
    if ($null -ne $knownProcess) {
        $workflowProcessIds = @([int]$knownProcess.ProcessId)
    }
    else {
        $workflowProcessIds = @(
            Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.CommandLine -like "*run_strict_1800_workflow.py*" -and
                    $_.CommandLine -like "*formal_1800_dual_v4*"
                } |
                ForEach-Object { $_.ProcessId }
        )
    }

    $latestResult = Get-ChildItem -LiteralPath $outputDir -Recurse -Filter result.json -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    $record = @{
        workflow_contract = $state.workflow_contract
        state_status = $state.status
        stage = $state.stage
        current = $state.current
        process_count = $workflowProcessIds.Count
        process_ids = $workflowProcessIds
        state_updated_at = $state.updated_at
        latest_result_path = if ($null -eq $latestResult) { $null } else { $latestResult.FullName }
        latest_result_written_at = if ($null -eq $latestResult) { $null } else { $latestResult.LastWriteTime.ToString("o") }
    }

    if ($workflowProcessIds.Count -gt 0) {
        Save-WorkflowPid -ProcessId ([int]$workflowProcessIds[0])
        $record.status = "healthy_running"
        Write-MonitorRecord $record
        exit 0
    }

    if ($state.status -in @("complete", "completed", "stopped_by_user")) {
        $record.status = "terminal_not_restarted"
        Write-MonitorRecord $record
        exit 0
    }

    $stateAgeMinutes = ((Get-Date) - (Get-Item -LiteralPath $statePath).LastWriteTime).TotalMinutes
    $resultAgeMinutes = if ($null -eq $latestResult) {
        [double]::PositiveInfinity
    }
    else {
        ((Get-Date) - $latestResult.LastWriteTime).TotalMinutes
    }
    $record.state_age_minutes = [math]::Round($stateAgeMinutes, 2)
    $record.latest_result_age_minutes = if ([double]::IsPositiveInfinity($resultAgeMinutes)) {
        $null
    }
    else {
        [math]::Round($resultAgeMinutes, 2)
    }

    if ([math]::Min($stateAgeMinutes, $resultAgeMinutes) -lt $staleActivityMinutes) {
        $record.status = "healthy_recent_activity_process_not_visible"
        Write-MonitorRecord $record
        exit 0
    }

    if (-not (Test-Path -LiteralPath $pythonPath)) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            $record.status = "python_missing"
            Write-MonitorRecord $record
            exit 2
        }
        $pythonPath = $pythonCommand.Source
    }

    if (-not (Test-Path -LiteralPath $workflowScript) -or -not (Test-Path -LiteralPath $webotsPath)) {
        $record.status = "required_executable_missing"
        $record.workflow_script_exists = Test-Path -LiteralPath $workflowScript
        $record.webots_exists = Test-Path -LiteralPath $webotsPath
        Write-MonitorRecord $record
        exit 2
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $resumeStdout = Join-Path $monitorDir "resume_${timestamp}.stdout.log"
    $resumeStderr = Join-Path $monitorDir "resume_${timestamp}.stderr.log"
    $arguments = @(
        "scripts\run_strict_1800_workflow.py",
        "--phase", "all",
        "--webots", ('"' + $webotsPath + '"'),
        "--output-dir", $outputDir,
        "--report-path", $reportPath
    )

    $resumed = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $arguments `
        -WorkingDirectory $RepositoryRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $resumeStdout `
        -RedirectStandardError $resumeStderr `
        -PassThru

    Save-WorkflowPid -ProcessId $resumed.Id
    $record.status = "restarted_from_checkpoint"
    $record.resumed_process_id = $resumed.Id
    $record.resume_stdout = $resumeStdout
    $record.resume_stderr = $resumeStderr
    Write-MonitorRecord $record
}
catch {
    Write-MonitorRecord @{
        status = "monitor_error"
        error = $_.Exception.Message
        stack = $_.ScriptStackTrace
    }
    exit 2
}
finally {
    if ($lockTaken) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
