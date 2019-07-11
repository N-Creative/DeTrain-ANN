<?php
require_once "DeTrain.php";

//Create the neural network with 2 inputs, 2 hidden layers with 3 neurons in each, and 1 output
$ann = new DeTrain([2, 3, 3, 1]);

//Samples
$xor = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];

echo "Training of test ANN..." . PHP_EOL;

for ($i = 0; $i < 1000; $i++) {
    foreach ($xor as $row) {
        $target = [floatval($row[0] xor $row[1])];
        $ann->train($row, $target, 0.1);
    }
}

echo "Testing ANN..." . PHP_EOL;

foreach ($xor as $row) {
    $t = ($row[0] xor $row[1]);
    $result = $ann->run($row)[0];
    echo $row[0] . " xor " . $row[1] . " = " . round($result, 8) . PHP_EOL;
}
