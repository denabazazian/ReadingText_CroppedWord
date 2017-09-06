# ReadingText_CroppedWord
- Reading the text of cropped word images

 1. Make a confidence for each cropped word
 2. Pass each confidence to the DictNet 
 3. Read the text of each cropped word

## Run

- Shell commands of running this code: <br />
** ``` python confidenceOfCropped.py /path/to/input/images/*.jpg ``` <br /> 
** ``` getDictNet.py -gpu=0 '-inputDir=/path/to/input/images/' conf2dictnet /path/to/confidence/*.csv ``` <br />
** ``` getDictNet.py dictnet2final ./vggtr_confidence/*.csv -vocDir=./voc_strong -dictnetThr=.4 ``` <br />
** ``` getDictNet.pyLTWHTr2icdar4pZip -outfile=./zips/finalDN40Iou30Vocstrong_vggtr_confidence.zip /finalDN40Iou30Vocstrong_vggtr_confidence/*.csv ``` <br />


## Citation
This pipeline was used for reading compressed cropped word image of this [paper](http://www.micc.unifi.it/seidenari/wp-content/papercite-data/pdf/iccv_epic_2017.pdf). Please cite this work in your publications if it helps your research: <br />
@article{Galteri17, <br />
author  = {Galteri, Leonardo and Bazazian, Dena and Seidenari, Lorenzo and Bertini, Marco and Bagdanov, Andrew D and Nicolau, Anguelos and Karatzas, Dimosthenis and Del Bimbo,  Alberto},<br />
title   = {Reading Text in the Wild from Compressed Images},<br />
journal = {ICCV (International Conference on Computer Vision), EPIC workshop},<br />
year    = {2017},<br />
ee      = {[link](http://www.micc.unifi.it/seidenari/wp-content/papercite-data/pdf/iccv_epic_2017.pdf)} <br />
}
