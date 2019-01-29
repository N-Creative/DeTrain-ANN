<?php
require_once "DeTrain.php";

//Create the neural network with 2 inputs, 3 hidden neurons and 1 output
$ann = new DeTrain(2, 3, 1);

//This array is learning set
$train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];

echo "Training of test ANN...\n";

$t = []; $output = [];
for ($i = 0; $i < 100; $i++)
    foreach ($train as $row) {
        $t[1] = ($row[0] xor $row[1]);
        $output = $ann->run($row, true);
        $ann->train($t, $output, 0.5);
    }

echo "Testing ANN...\n";

foreach ($train as $row) {
    $output = $ann->run($row);
    $result = round($output[1], 3);
    echo "$row[0] xor $row[1] = $output\n";
}
