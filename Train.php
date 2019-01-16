<?php
/*
 * © N-Creative, 2019. Доступно по лицензии MIT.
 */

//TODO: Логические ошибки в работе нейронной сети, нужно найти их

class Train {
    protected $nInput; //кол-во входов
    protected $nHidden; //кол-во ассоциативных нейронов

    protected $input = [];
    protected $hidden = [];
    protected $output;

    protected $weightSA = []; //веса от входа к скрытому слою
	protected $weightAR = []; //веса от скрытого слоя к выходу

    public function __construct(int $in, int $hdn) {
        $this->nInput = $in;
        $this->nHidden = $hdn;

        //матрица весов: строки - нейроны текущего слоя, столбцы - нейроны следующего слоя
        $this->weightAR[0] = rand(-5000, 5000) / 10000;
        for ($h = 1; $h <= $hdn; $h++) {
            for ($i = 0; $i <= $in; $i++)
                $this->weightSA[$i][$h] = rand(-5000, 5000) / 10000; //инициализация весов SA

            $this->weightAR[$h] = rand(-5000, 5000) / 10000; //инициализация весов AR
        }
    }

    //Запуск нейронной сети
    public function run(array $data, bool $train = false) {
        $nInput = &$this->nInput;
        $nHidden = &$this->nHidden;

        if (count($data) != $nInput) return false;

        //распространение сигналов по SA-дендритам
        $hidden = [];
        for ($h = 1; $h <= $nHidden; $h++) {
            $hidden[$h] = $this->weightSA[0][$h];
            for ($i = 1; $i <= $nInput; $i++)
                $hidden[$h] += $data[$i - 1] * $this->weightSA[$i][$h];
        }

        //распространение сигналов по AR-дендритам
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

    //DEBUG ME PLS!!!
    //обучение методом обратного распространения ошибки
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

        //распространение от выхода к скрытым слоям
        $do = ($t - $output) / cosh($this->output);
        $dwAR[0] = $a * $do;
        for ($h = 1; $h <= $nHidden; $h++)
            $dwAR[$h] = $a * $do * $this->sigmoid($hidden[$h]);        

        //распространение от скрытых слоёв ко входу
        for ($h = 1; $h <= $nHidden; $h++) {
            $da[$h] = $do * $weightAR[$h] / cosh($hidden[$h]);
            $dwSA[0][$h] = $a * $da[$h];
            for ($i = 1; $i <= $nInput; $i++)
                $dwSA[$i][$h] = $a * $da[$h] * $input[$i - 1];           
        }

        //правим веса
        for ($h = 1; $h <= $nHidden; $h++)
            for ($i = 0; $i <= $nInput; $i++)
                $weightSA[$i][$h] += $dwSA[$i][$h];

        for ($h = 0; $h <= $nHidden; $h++)
            $weightAR[$h] += $dwAR[$h];
    }

    //Сигмоида
    protected function sigmoid(float $x) {
        return (2 / (1 + exp(-$x)) - 1);
    }
}
