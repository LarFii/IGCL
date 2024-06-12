# IGCL

## Requirement

* python == 3.7.7
* pytorch == 1.9.1
* numpy == 1.20.3
* scipy == 1.7.1
* pandas == 1.3.4
* cython == 0.29.24

## Start

* Step 1

```python
python local_compile_setup.py build_ext --inplace
```
* Step 2

```python
python main.py --recommender=IGCL --dataset=tmall --ssl_reg=0.4 --ssl_ratio=0.4 --ssl_temp=0.2 --recon_reg=2 --IGCL_layers=3 --recon_dim=1024 --rnoise_eps=0.8
```
