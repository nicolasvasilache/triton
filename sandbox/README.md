```
python sandbox/gluon_memcpy_cross_compile.py
python sandbox/gluon_memcpy_benchmark.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python sandbox/dist_sync.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python sandbox/gluon_memcpy_dist_benchmark.py
```