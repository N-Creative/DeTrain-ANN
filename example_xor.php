<?php
require_once "Train.php";

//Create the neural network with 2 inputs and 3 hidden neurons
$ann = new Train(2, 3);

//This array is learning set
$train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];

echo "Training of test ANN...\n";

for ($i = 0; $i < 2000; $i++)
    foreach ($train as $row) {
        $t = ($row[0] xor $row[1]);
        $output = $ann->run($row, true, 2);
        $ann->train($t, $output);
    }

echo "Testing ANN...\n";

foreach ($train as $row) {
    $output = round($ann->run($row), 3);
    echo "$row[0] xor $row[1] = $output\n";
}
