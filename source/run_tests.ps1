param(
    [switch]$BigCrush = $false
)

$SEED = 12110759528222028398
$IMAGE_NAME = "marzopa/prng-bench"
$RESULTS_DIR = "${PWD}/results"

if (!(Test-Path $RESULTS_DIR)) { New-Item -ItemType Directory -Path $RESULTS_DIR }

function Launch-Test ($name, $algo, $delta) {
    $fullName = "${name}_${SEED}"

    if ($BigCrush){
        $reportFile = "/app/results/${fullName}_BigCrush.txt"
        $binary = "./test_from_pipe_BigCrush"
    }
    else{
        $reportFile = "/app/results/${fullName}_Crush.txt"
        $binary = "./test_from_pipe"
    }

    $cmd = "python3 testing_interface.py p $SEED $delta --algo $algo | $binary > $reportFile"

    Write-Host "Launching $fullName... (Output: $reportFile)"

    $containerName = "${fullName}_run"

    docker rm -f $containerName 2>$null | Out-Null

    docker run -d --rm --name $containerName `
        -v "${PWD}/results:/app/results" `
        $IMAGE_NAME `
        sh -c $cmd
}

Write-Host "--- Starting Parallel Tests ---" -ForegroundColor Cyan

Launch-Test "Lcomp" "Lcomp" 0
Launch-Test "Hcomp" "Hcomp" 0

Write-Host "Tests launched! Check the 'results' folder for active logs." -ForegroundColor Yellow