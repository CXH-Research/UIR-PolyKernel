# UIR-PolyKernel

[Underwater Image Restoration via Polymorphic Large Kernel CNNs](https://arxiv.org/abs/2412.18459)
<div>
<span class="author-block">
  Xiaojiao Guo<sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    Yihang Dong<sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    <a href='https://cxh.netlify.app/'>Xuhang Chen</a>
  </span>,
  <span class="author-block">
    Weiwen Chen
  </span>,
  <span class="author-block">
    Zimeng Li<sup> 📮</sup>
  </span>,
  <span class="author-block">
    <a href='https://lzeeorno.github.io/'>FuChen Zheng</a>
  </span>,
  <span class="author-block">
    <a href='https://cmpun.github.io/'>Chi-Man Pun</a><sup> 📮</sup>
  </span>
  ( 👨‍💻‍ Equal contributions, 📮 Corresponding author)
</div>

<b>University of Macau, SIAT CAS, Huizhou Univeristy, Shenzhen Polytechnic University, The Hong Kong University of Science and Technology (Guangzhou), Baoshan Univeristy</b>

In <b>_IEEE International Conference on Acoustics, Speech, and Signal Processing 2025 (ICASSP 2025)_</b>

# ⚙️ Usage

## Training
You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

## Inference

Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.

```bash
python test.py
```

# Citation

```bib
@inproceedings{guo2025underwater,
  title={Underwater Image Restoration via Polymorphic Large Kernel CNNs},
  author={Guo, Xiaojiao and Dong, Yihang and Chen, Xuhang and Chen, Weiwen and Li, Zimeng and Zheng, FuChen and Pun, Chi-Man},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
