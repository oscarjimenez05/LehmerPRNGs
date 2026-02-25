param(
    [switch]$BigCrush = $false
)

$SEED0 = 12229249494144361908
$SEED1 = 8816499587629192363
$SEED2 = 305445197473875291
$SEED3 = 7412461098979201985
$SEED4 = 13485151628806994
$IMAGE_NAME = "marzopa/prng-bench"

$RESULTS_DIR = "${PWD}/results"
if (!(Test-Path $RESULTS_DIR)) { New-Item -ItemType Directory -Path $RESULTS_DIR }

function Launch-Test ($name, $delta) {
    $fullName = "${name}_${SEED0}_${SEED1}_${SEED2}_${SEED3}_${SEED4}"

    if ($BigCrush){
        $reportFile = "crypto/results/${fullName}_BigCrush.txt"
        $binary = "./test_from_pipe_BigCrush"
    }
    else{
        $reportFile = "crypto/results/${fullName}_Crush.txt"
        $binary = "./test_from_pipe"
    }

    $cmd = "python3 crypto/crypto_testing_interface.py $SEED0 $SEED1 $SEED2 $SEED3 $SEED4--delta $delta | $binary > $reportFile"

    Write-Host "Launching $fullName... (Output: $RESULTS_DIR\${fullName}_...)"

    $containerName = "${fullName}_run"

    docker rm -f $containerName 2>$null | Out-Null

    docker run -d --rm --name $containerName `
        -v "${RESULTS_DIR}:/app/crypto/results" `
        $IMAGE_NAME `
        sh -c $cmd
}

Write-Host "--- Starting CryptoLehmer TestU01 Tests ---" -ForegroundColor Cyan

Launch-Test "CryptoLehmer_d0" 0 BigCrush

Write-Host "Tests launched! Check the 'results' folder for logs." -ForegroundColor Yellow