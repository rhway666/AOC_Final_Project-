# The main code for model analysis
First activate the conda environment aoc_final
## Downloading the DeiT Tiny Distilled Model
The pre-trained model `deit_tiny_distilled-patch16-224` is not included in this repository due to its size. To download and use it, follow these steps:

1. **Download the Model**:  
   You can download the `deit_tiny_distilled-patch16-224` model from the official Hugging Face Model Hub or the original source:
   - Hugging Face: [DeiT Tiny Distilled](https://huggingface.co/facebook/deit-tiny-distilled-patch16-224)
   - Alternatively, use the `transformers` library to download it automatically (see code below).
2. Or use the download model script
```
python3 download_model.py
```
## Analysis 

For Model Architecture and shape
```
python3 shape.py
python3 print_model_arch.py
```
We provide CPU Profiling for DeiT Tiny Model
```
# for full model
python3 proiling_whoie_model.py

# for encoder profiling
python3 encoder_profile.py

```