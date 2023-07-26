# Nuclei annotation
performing nuclei annotation (manually or semi-automatically) using ImageJ

## Table of Contents 
[Citation](#citation)

[Manual instance segmentation with ImageJ](#manual-instance-segmentation-with-imagej)

[Manual instance segmentation and classification with ImageJ](#manual-instance-segmentation-and-classification-with-imagej)


[Semiautomatic annotation with ImageJ](#semiautomatic-annotation-with-imagej)

[Codes to generate segmentation masks](#codes-to-generate-segmentation-masks)



## Citation
The instructions used in this repository are adapted from these articles. If you use them, please cite the following articles:

BibTex entries:
```
@article{CryoNuSeg2021,
title = "{CryoNuSeg}: A dataset for nuclei instance segmentation of cryosectioned H\&E-stained histological images",
journal = "Computers in Biology and Medicine",
volume = "132",
pages = "104349",
year = "2021",
doi = "https://doi.org/10.1016/j.compbiomed.2021.104349",
author = "Amirreza Mahbod and Gerald Schaefer and Benjamin Bancher and Christine L\"{o}w and Georg Dorffner and Rupert Ecker and Isabella Ellinger"
}
```

```
@article{doi:10.1091/mbc.E20-02-0156,
title = "{AnnotatorJ}: an ImageJ plugin to ease hand annotation of cellular compartments",
journal = "Molecular Biology of the Cell",
volume = "31",
number = "20",
pages = "2179-2186",
year = "2020",
doi = "10.1091/mbc.E20-02-0156",
author = "Hollandi, R\'{e}ka and Di\'{o}sdi, \'{A}kos and Hollandi, G\'{a}bor and Moshkov, Nikita and Horv\'{a}th, P\'{e}ter"
}
```


## Manual instance segmentation with ImageJ
We used ImageJ software to perform manual nuclei instance segmentation. We followed these steps to annotate the images manually:
- Download and open the software (ImageJ-win64.exe), download from Fiji: https://imagej.net/software/fiji/downloads
- Update the software for the first time (help --> update)
- Open an image with the software
- From tabs:  Analyse --> Tools --> ROI manager. Ensure that both "show all" and "labels" are activated in the ROI manager. 
- Zoom in/out to have a clear view of the image and all instances
- From the selection options, select "freehand selection"
- Manually annotate the border for each object and press "T". To remove an object select the labeled number inside the object and then press "Delete" to remove an ROI 
- When you are done with all nuclei, save the outputs with ROI manager--> More --> Save
- A zip file containing all ROI files will be created after saving the outputs (each ROI file represent one of the nuclei)
- The created zip file can be later processed with Matlab (create labeled masks, binary masks, etc)

## Manual instance segmentation and classification with ImageJ
Consider that we have two classes. The following instructions can be easily extended in case of more nuclei classes.
- Download and open the software (ImageJ-win64.exe), download from Fiji: https://imagej.net/software/fiji/downloads
- Update the software for the first time (help --> update)
- Click on ">>" to add "ROI Menu" to the main software tabs
- Open an image with the software
- From the selection options, select "freehand selection"
- From tabs:  Analyse --> Tools --> ROI manager. Ensure that both "show all" and "labels" are activated in the ROI manager.
  
  #### For class one: ROI -->set default group (0 represents yellow )
- Zoom in/out to have a clear view of the image and all instances
- Manually annotate the border for each object and press "T". To remove an object select the labeled number inside the object and then press "Delete" to remove an ROI
  
  #### For class two: ROI -->set default group (2 represents red )
- Zoom in/out to have a clear view of the image and all instances
- Manually annotate the border for each object and press "T". To remove an object select the labeled number inside the object and then press "Delete" to remove an ROI
  
- When you are done with all nuclei for both classes, then:<br>
1- ROI --> select group 0 --> save the outputs with ROI manager--> More --> Save (create a zip file with instances of class 1)<br>
2- ROI --> select group 1 --> save the outputs with ROI manager--> More --> Save (create a zip file with instances of class 2)<br>
3- Select all instances   --> save the outputs with ROI manager--> More --> Save (create a zip file with instances of all classes)

Note: 
- The class of instance can be changed using the "properties" option
- Other numbers for each class can be selected (each number represents a color e.g. number 3 for green)



## Semiautomatic annotation with ImageJ


## Codes to generate segmentation masks
The codes are available on the CryoNuSeg repository: https://github.com/masih4/CryoNuSeg/tree/master 
