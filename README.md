# CompactBilinearPooling-Pytorch

A Pytorch Implementation for Compact Bilinear Pooling. Adapted from [tensorflow_compact_bilinear_pooling](https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling)


## Prerequisites

Install [pytorch_fft](https://github.com/locuslab/pytorch_fft) by 

```bash
pip install pytorch_fft
```



## Usage

```python
from torch import nn
from torch.autograd import Variable
from CompactBilinearPooling import CompactBilinearPooling

bottom1 = Variable(torch.randn(128, 512, 14, 14)).cuda()
bottom2 = Variable(torch.randn(128, 512, 14, 14)).cuda()

layer = CompactBilinearPooling(512, 512, 8000)
layer.cuda()
layer.train()

out = layer(bottom1, bottom2)
```



## Reference

```
Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).
```



