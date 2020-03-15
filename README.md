# GridTorch

This repo contains a pytorch implementation of the supervised half of Banino et al's 2018 Nature paper: 
[<em>Vector-based navigation using grid-like representations in artificial agents.</em>](https://www.nature.com/articles/s41586-018-0102-6).

## Dataset

Original implementation and DATA can be found here:
https://github.com/deepmind/grid-cells

You need to first download [original dataset](https://console.cloud.google.com/storage/browser/grid-cells-datasets) to `../data/`. 

Then run `python convert_data.py` to convert tensorflow data to pytorch format.

The converted pytorch version dataset could be downloaded [here](https://drive.google.com/file/d/1rWLDdbtszSGV1sDJdNlUjR987-XVSzd7/view?usp=sharing).
