# Effective Transmission and Classification of Compressed Wildfire Images Over Noisy Channels with Memory

Simple overview of use/purpose.

## Description

An in-depth paragraph about your project and overview of use.

## Project Structure

The projects directory structure can be seen below. 

* scalar_quantization
    * cosq_rate_2_G.ipynb
    * cosq_rate_2_L.ipynb
    * cosq_rate_3_G.ipynb
    * cosq_rate_4_G.ipynb
* vector_quantization
    * vector_train
        * covq_design.py
        * main_covq_train.py
        * main_vq_train.py
    * vector_test
        * decoder.py
        * encoder.py
        * main.py
    * vector_codebooks
        * covq
        * vq
    * vector_utils
* conv_neural_network
    * nn_eval.py
    * nn_train.py
* web_app
* images
    * test
        * misc_images
        * nowildfire
        * wildfire
    * train
        * misc_images
        * nowildfire
        * wildfire

## Getting Started

### Executing program

* Vector Quantization
    * Run the following command in the terminal from the root directory.
    ```
    python3 -m vector_quantization.vector_test.main
    ``` 
    * The function allows for four parameter inputs: _k_, _n_, _epsilon_ and _image_.
        * _k_: Vector dimension size. Allowable values in {4, 16, 64}
        * _n_: Number of codewords. Allowable values in {4, 8, 16, 32, 64, 128, 256}
        * _epsilon_: Noise percentage. Allowable values in {0, 0.005, 0.01, 0.05, 0.1}
        * _image_: File name of image. Image must be in images/test/misc_images/ directory
    * Run the following command to specify parameters. 
    ```
    python3 -m vector_quantization.vector_test.main --k 16 --n 256 --epsilon 0.1 --image 'satelliteIMG02.jpg'
    ``` 
    * Default parameters are those seen in the command above.


## Authors

The team consists of the following members: 
* [Taylor Balsky](https://github.com/taylorbalsky)
* [Adam Cormier](https://github.com/adamcorm28)
* [Campbell Harris](https://github.com/CampbellHarris02)
* [Xiaoyu (Sean) Ren](https://github.com/SeanRen01)
* [Nigel Vasoff](https://github.com/Lques)


## Acknowledgements

The team would like to thank Dr. Fady Alajaji for assisting with the project.