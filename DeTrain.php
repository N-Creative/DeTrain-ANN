<?php
/*
 * © N-Creative, 2019. This code is available under the MIT license.
 */

define("RAND", 500000000);

class DeTrain {
    protected $nInput; //count of inputs
    protected $nHidden; //count of hidden neurons
    protected $nOutput; //count of outputs

    protected $input = [];
    protected $hidden = [];
    protected $output = [];

    protected $weightSA = []; //weights from input to hidden neurons
    protected $weightAR = []; //weights from hidden neurons to output

    public function __construct(int $in, int $hdn, int $out) {
        $this->nInput = $in;
        $this->nHidden = $hdn;
	$this->nOutput = $out;

	//the matrix of weights: rows is neurons of current layer, columns is neurons of next layer
	for ($i = 0; $i <= $in; $i++)
            for ($h = 1; $h <= $hdn; $h++)
                $this->weightSA[$i][$h] = rand(-RAND, RAND) / (2 * RAND); //init SA weights

        for ($h = 0; $h <= $hdn; $h++)
            for ($o = 1; $o <= $out; $o++)
                $this->weightAR[$h][$o] = rand(-RAND, RAND) / (2 * RAND); //init AR weights
    }

    //Run the neural network
    public function run(array $data, bool $train = false) {
        $nInput = &$this->nInput;
        $nHidden = &$this->nHidden;

        if (count($data) != $nInput) return false;

        //SA forward propagation
        $hidden = [];
        for ($h = 1; $h <= $nHidden; $h++) {
            $hidden[$h] = $this->weightSA[0][$h];
            for ($i = 1; $i <= $nInput; $i++)
                $hidden[$h] += $data[$i - 1] * $this->weightSA[$i][$h];
        }

	//AR forward propagation
        $output = [];
        for ($o = 1; $o <= $nOutput; $o++) {
            $output[$o] = $this->sigmoid($this->weightAR[0][$o]);
            for ($h = 1; $h <= $nHidden; $h++)
                $output[$o] += $this->sigmoid($hidden[$h]) * $this->weightAR[$h][$o];
        }

        if ($train) {
            $this->input = $data;
            $this->hidden = $hidden;
            $this->output = $output;
        }

        return $this->sigmoid($output);
    }

    //Back propagation training
    public function train(float $t, float $output, float $a = 1) {
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
        for ($o = 1; $o <= $nOutput; $o++) {
            $dwAR[0][$o] = $a * ($t - $output[$o]) * cosh($this->output[$o]);
            for ($h = 1; $h <= $nHidden; $h++)
                $dwAR[$h][$o] = $dwAR[0][$o] * $this->sigmoid($hidden[$h]);  
        }       

        //propagation from hidden layer to input
        for ($h = 1; $h <= $nHidden; $h++) {
            $dwSA[0][$h] = 0;
            for ($o = 1; $o <= $nOutput; $o++)
                $dwSA[0][$h] += $dwAR[0][$o] * $weightAR[$h][$o];

            $dwSA[0][$h] *= cosh($hidden[$h]);

            for ($i = 1; $i <= $nInput; $i++)
                $dwSA[$i][$h] = $dwSA[0][$h] * $input[$i - 1];           
        }

        //edit the weights
        for ($h = 1; $h <= $nHidden; $h++)
            for ($i = 0; $i <= $nInput; $i++)
                $weightSA[$i][$h] += $dwSA[$i][$h];

        for ($o = 1; $o <= $nOutput; $o++)
            for ($h = 0; $h <= $nHidden; $h++) 
                $weightAR[$h][$o] += $dwAR[$h];
    }

    protected function sigmoid(float $x) {
        return (2 / (1 + exp(-$x)) - 1);
    }
}
