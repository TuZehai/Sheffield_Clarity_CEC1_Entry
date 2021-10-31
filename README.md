# Implementation of the Sheffield entry for the first Clarity enhancement challenge (CEC1)
This repository contains the PyTorch implementation of "[A Two-Stage End-to-End System for Speech-in-Noise Hearing Aid Processing](https://claritychallenge.github.io/clarity2021-workshop/papers/Clarity_2021_paper_tu.pdf)", the Sheffield entry for the first [Clarity enhancement challenge (CEC1)](https://github.com/claritychallenge/clarity_CEC1/). The system consists of a [Conv-TasNet](https://github.com/kaituoxu/Conv-TasNet) based denoising module, and a finite-inpulse-response (FIR) filter based amplification module. A differentiable approximation to the [Cambridge MSBG model](https://github.com/claritychallenge/clarity_CEC1/tree/master/projects/MSBG) released in the CEC1 is used in the loss function.

## Requirements
To run the training recipe of the amplification module, [the MSBG package](https://github.com/claritychallenge/clarity_CEC1/tree/master/projects/MSBG) and [PyTorch STOI](https://github.com/mpariente/pytorch_stoi) are needed.

## Training
To build the overall system, the Conv-TasNet based denoising module needs to be trained in the first stage, and the scripts are in the recipe_den_convtasnet. The FIR based amplification module is trained in the second stage, and the scripts are in the recipe_amp_fir. The MBSTOI folder contains the MBSTOI implementation from [the CEC1 project](https://github.com/claritychallenge/clarity_CEC1/tree/master/projects/MBSTOI), with also the DBSTOI implementation.

## References
* [1] Luo Y, Mesgarani N. Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation[J]. IEEE/ACM transactions on audio, speech, and language processing, 2019, 27(8): 1256-1266.
* [2] Andersen A H, de Haan J M, Tan Z H, et al. Refinement and validation of the binaural short time objective intelligibility measure for spatially diverse conditions[J]. Speech Communication, 2018, 102: 1-13.
* [3] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech', ICASSP 2010, Texas, Dallas.

## Citation
If you use this work, please cite:
```
@article{tutwo,
  title={A Two-Stage End-to-End System for Speech-in-Noise Hearing Aid Processing},
  author={Tu, Zehai and Zhang, Jisi and Ma, Ning and Barker, Jon},
  year={2021},
  booktitle={The Clarity Workshop on Machine Learning Challenges for Hearing Aids (Clarity-2021)},
}
```
