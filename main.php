<?php
require_once "Train.php";

$ann = new Train(2, 3);

$train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];

echo "Training of test ANN...\n";

for ($i = 0; $i < 500; $i++)
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