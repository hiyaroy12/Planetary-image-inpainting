## Planetary-image-inpainting
Planetary image inpainting by learning modality specific regression models.

### Introduction:
Sophisticated imaging instruments on-board spacecraft orbiting different planets and planetary bodies in this solar system, enable humans to discover and visualize the unknown. However, these planetary surface images suffer from some missing pixel regions, which could not be captured by the spacecraft onboard cameras because of some technical limitations. In this work, we try to inpaint these missing pixels of the planetary images using modality-specific regression models that were trained with clusters of different images with similar histogram distribution on the experimental dataset. Filling in missing data via image inpainting enables downstream scientific analysis such as the analysis of morphological features on the planetary surface - e.g., craters and their sizes, central peaks, interior structure, etc|in comparison with other planetary bodies. Here, we use the grayscale version of Mars orbital images captured by the HiRISE (High-Resolution Imaging Science Experiment) camera on the Mars Reconnaissance Orbiter (MRO) for the experimental purpose. The results show that our method can fill in the missing pixels existing on the Mars surface image with good visual and perceptual quality and improved PSNR values. Detailed description of the system can be found in our [paper.](https://www.dropbox.com/s/fkb148zciaj69r6/elsarticle_final.pdf?dl=0)

## Dataset: 
- Please download the dataset from here: [Mars orbital image (HiRISE)labeled data set](https://zenodo.org/record/2538136#.XYjEuZMzagR)

### 1) Training
To train the model, run:
```bash
python main.py
```
