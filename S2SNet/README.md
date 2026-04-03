# S2SNet

This is the repo for S2SNet: A Pretrained Neural Network for Superconductivity Discovery

If you use our work then please cite

```
@inproceedings{ijcai2022p708,
  title     = {S2SNet: A Pretrained Neural Network for Superconductivity Discovery},
  author    = {Liu, Ke and Yang, Kaifan and Zhang, Jiahong and Xu, Renjun},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {5101--5107},
  year      = {2022},
  month     = {7},
  doi       = {10.24963/ijcai.2022/708},
  url       = {https://doi.org/10.24963/ijcai.2022/708},
}
```

Superconductivity allows electrical current to flow without any energy loss, and thus making solids superconducting is a grand goal of physics, material science, and electrical engineering. More than 16 Nobel Laureates have been awarded for their contribution in superconductivity research. Superconductors are valuable  for sustainable development goals (SDGs), such as climate change mitigation, affordable and clean energy, industry, innovation and infrastructure, and so on. However, a unified physics theory explaining all superconductivity mechanism is still unknown. It is believed that superconductivity is microscopically due to not only molecular compositions but also the geometric crystal structure. Hence a new dataset, S2S, containing both crystal structures and superconducting critical temperature, is built upon SuperCon and Material Project. Based on this new dataset, we propose a novel model, S2SNet, which utilizes the attention mechanism for superconductivity prediction. To overcome the shortage of data, S2SNet is pre-trained on the whole Material Project dataset with Masked-Language Modeling (MLM). S2SNet makes a new state-of-the-art, with out-of-sample accuracy of 92\% and Area Under Curve (AUC) of 0.92.

Paper link: waiting for update
Pre-traind models can be found on [Google Drive](https://drive.google.com/drive/folders/1jhREmq4VZC2U-xEYWeni1oD5U-4f6KmV?usp=sharing)

Code Author: [Ke Liu](https://github.com/zjuKeLiu/S2SNet) & [Kaifan Yang](https://github.com/ykfreborn)
