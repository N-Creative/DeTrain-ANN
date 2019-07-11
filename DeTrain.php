<?php
/*
 * DeTrain.php
 * A multilayer perceptron
 * © N-Creative, 2019. This code is available under the MIT license.
 */

//Under this comment you can insert a namespace;

define("RAND", 1000000000);
define("RAND_1_2", 500000000);
define('M_FI', 1.61803398874989);

class DeTrain {
    protected $layerMap = [];       //map of layers, contains counts of neurons in every layer
    protected $neuronVals = [];     //
    protected $weights = [];        //$weights[$l][$c][$n], where $l - current layer, $c - current neuron number [0..n] (0 - offset), $n - next neuron number [1..n]

    public function __construct(array $layerMap) {
        $layers = count($layerMap);
        if ($layers < 3) {
            throw new Error("Required 3 layers at least");
        }

        $layers--;
        $this->layerMap = $layerMap;
        for ($l = 0; $l < $layers; $l++) {
            $this->initMatrix($l);
        }
    }

    protected function initMatrix(int $layer) {
        $curLength = $this->layerMap[$layer];
        $nextLength = $this->layerMap[$layer + 1];

        for ($cur = 0; $cur <= $curLength; $cur++) {
            for ($next = 1; $next <= $nextLength; $next++) {
                $this->weights[$layer][$cur][$next] = rand(-RAND, RAND) / RAND_1_2;
            }
        }
    }

    public static function load(string $name = "core") : DeTrain {
        $path = __DIR__ . "/$name.dtn";
        
        if (file_exists($path))
            return unserialize(file_get_contents($path));
        else
            return null;
    }

    public function run(array $data, bool $train = false) {
        $map = $this->layerMap;

        if (count($data) != $map[0]) {
            throw new Error("Length of input data array doesn't match to count of inputs of the ANN.");
        }

        $layers = count($map) - 1;

        $curow = $this->fromOne($data);
        $this->neuronVals[0] = $curow;
        for ($l = 0; $l < $layers; $l++) {
            $curow = $this->handlePropagation($curow, $l, $train);
        }

        return $this->fromZero($curow); // $this->f($curow)
    }

    protected function handlePropagation(array $curow, int $l, bool $train) {
        $map = $this->layerMap;
        
        $nextrow = [];
        for ($next = 1; $next <= $map[$l + 1]; $next++) {
            $buf = $this->weights[$l][0][$next];
            for ($cur = 1; $cur <= $map[$l]; $cur++) {
                $addend = $this->sigmoid($curow[$cur]);
                $buf += $addend * $this->weights[$l][$cur][$next];
            }

            $nextrow[$next] = $buf;
        }

        if ($train) {
            $this->neuronVals[$l + 1] = $nextrow;
        }
        
        return $nextrow;
    }

    public function train(array $input, array $target, float $a = 0.1) : void {
        $output = $this->run($input, true);

        $dW = [];
        $map = $this->layerMap;
        $layers = count($map) - 1;
        $preLast = $layers - 1;

        //from output layer to last hidden layer
        for ($o = 1; $o <= $map[$layers]; $o++) {
            $left = $o - 1;
            $buf = $a * ($target[$left] - $output[$left]); // * $this->dSigmoid($outIn[$o]);
            $dW[$preLast][0][$o] = $buf;
            for ($h = 1; $h <= $map[$preLast]; $h++) {
                $dW[$preLast][$h][$o] = $buf * $this->sigmoid($this->neuronVals[$preLast][$h]);
            }
        }        

        //from hidden layers to input
        for ($l = $preLast; $l > 0; $l--) {
            $pre = $l - 1;
            for ($h = 1; $h <= $map[$l]; $h++) {
                $buf = 0;
                for ($o = 1; $o <= $map[$l + 1]; $o++) {
                    $buf += $dW[$l][0][$o] * $this->weights[$l][$h][$o];
                }
    
                $buf *= $this->dSigmoid($this->neuronVals[$l][$h]);
                
                $dW[$pre][0][$h] = $buf;
                for ($i = 1; $i <= $map[$pre]; $i++) {
                    $dW[$pre][$i][$h] = $buf * $this->neuronVals[$pre][$i];
                }
            }
        }

        //edit the weights
        for ($l = 0; $l < $layers; $l++) {
            $next = $l + 1;

            for ($j = 1; $j <= $map[$next]; $j++) {
                for ($i = 0; $i <= $map[$l]; $i++) {
                    $this->weights[$l][$i][$j] += $dW[$l][$i][$j];
                }
            }        
        }
    }

    public function save($name = "core") : int {
        return file_put_contents(__DIR__ . "/$name.dtn", serialize($this));
    }

    protected function sigmoid(float $x) : float {
        return (2 / (1 + exp(-$x)) - 1);
    }

    protected function dSigmoid(float $x) : float {
        return ((1 - $this->sigmoid($x) ** 2) / 2);
    }

    protected function fromOne(array $input) : array {
        $result = [];
        foreach ($input as $i => $val) {
            $result[$i + 1] = $val;
        }
        return $result;
    }

    protected function fromZero(array $input) : array {
        $result = [];
        foreach ($input as $val) {
            $result[] = $val;
        }
        return $result;
    }
}
