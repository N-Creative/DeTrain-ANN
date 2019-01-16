<?php
/*
 * © N-Creative, 2019. This code is available under the MIT license.
 */

class Train {
    protected $nInput; //count of inputs
    protected $nHidden; //count of hidden neurons

    protected $input = [];
    protected $hidden = [];
    protected $output;

    protected $weightSA = []; //weights from input to hidden neurons
    protected $weightAR = []; //weights from hidden neurons to output

    public function __construct(int $in, int $hdn) {
        $this->nInput = $in;
        $this->nHidden = $hdn;

	//the matrix of weights: rows is neurons of current layer and columns is neurons of next layer
        $this->weightAR[0] = rand(-5000, 5000) / 10000;
        for ($h = 1; $h <= $hdn; $h++) {
            for ($i = 0; $i <= $in; $i++)
                $this->weightSA[$i][$h] = rand(-5000, 5000) / 10000; //init SA weights

            $this->weightAR[$h] = rand(-5000, 5000) / 10000; //init AR weights
        }
    }

    //Run the neural network
    public function run(array $data, bool $train = false) {
        $nInput = &$this->nInput;
        $nHidden = &$this->nHidden;

        if (count($data) != $nInput) return false;

        //SA propagation
        $hidden = [];
        for ($h = 1; $h <= $nHidden; $h++) {
            $hidden[$h] = $this->weightSA[0][$h];
            for ($i = 1; $i <= $nInput; $i++)
                $hidden[$h] += $data[$i - 1] * $this->weightSA[$i][$h];
        }

	//AR propagation
        $output = $this->sigmoid($this->weightAR[0]);
        for ($h = 1; $h <= $nHidden; $h++)
            $output += $this->sigmoid($hidden[$h]) * $this->weightAR[$h];

        if ($train) {
            $this->input = $data;
            $this->hidden = $hidden;
            $this->output = $output;
        }

        return $this->sigmoid($output);
    }

    //Back propagation training
    public function train(float $t, float $output, float $a = null) {
        if ($a === null) $a = 1;

        $nInput = &$this->nInput;
        $nHidden = &$this->nHidden;

        $input = &$this->input;
        $hidden = &$this->hidden;

        $weightSA = &$this->weightSA;
        $weightAR = &$this->weightAR;

        $dwAR = [];
        $dwSA = [];
        $da = [];

        //propagation from output layer to hidden layer
        $do = ($t - $output) / cosh($this->output); //derivative of sigmoid.
        $dwAR[0] = $a * $do;
        for ($h = 1; $h <= $nHidden; $h++)
            $dwAR[$h] = $a * $do * $this->sigmoid($hidden[$h]);        

        //propagation from hidden layer to input
        for ($h = 1; $h <= $nHidden; $h++) {
            $da[$h] = $do * $weightAR[$h] / cosh($hidden[$h]);
            $dwSA[0][$h] = $a * $da[$h];
            for ($i = 1; $i <= $nInput; $i++)
                $dwSA[$i][$h] = $a * $da[$h] * $input[$i - 1];           
        }

        //edit the weights
        for ($h = 1; $h <= $nHidden; $h++)
            for ($i = 0; $i <= $nInput; $i++)
                $weightSA[$i][$h] += $dwSA[$i][$h];

        for ($h = 0; $h <= $nHidden; $h++)
            $weightAR[$h] += $dwAR[$h];
    }

    protected function sigmoid(float $x) {
        return (2 / (1 + exp(-$x)) - 1);
    }
}
