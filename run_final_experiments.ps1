param(
    [string]$PythonExe = "py",
    [string]$Dataset = "flickr30k_hf",
    [string]$Split = "test",
    [string]$Device = "cpu",
    [int]$MaxExamples = 0,
    [string]$ExperimentSuffix = ""
)

$commonArgs = @(
    "run_experiment.py",
    "--dataset", $Dataset,
    "--split", $Split,
    "--device", $Device
)

if ($MaxExamples -gt 0) {
    $commonArgs += @("--max-examples", $MaxExamples.ToString())
}

$experiments = @(
    @{ Name = "baseline"; Args = @() },
    @{ Name = "prompt_photo_image"; Args = @("--prompt-type", "photo_image") },
    @{ Name = "prompt_detailed"; Args = @("--prompt-type", "detailed_scene") },
    @{ Name = "rerank_caption_prior"; Args = @("--rerank", "caption_prior", "--rerank-k", "25") }
)

foreach ($experiment in $experiments) {
    $experimentName = $experiment.Name
    if ($ExperimentSuffix) {
        $experimentName = "${experimentName}_${ExperimentSuffix}"
    }
    Write-Host "Running $experimentName..."
    & $PythonExe @commonArgs "--experiment-name" $experimentName @($experiment.Args)
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
