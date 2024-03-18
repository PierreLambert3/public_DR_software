"# public_DR_software" 

This is a data visualisation software centered around the use of dimensionality reduction. Multiple quality assessment (QA) methods are proposed for the embeddings, as well as a couple of dimensionality reduction (DR) methods. The pdf file brings a brief introduction to dimensionality reduction for data visualisation and details the various features of the software. The pdf's annexes describe the procedure to add your own datasets, QA criteria, or DR algorithms.
The software is not designed for very large datasets as the full distance matrix is computed in the background. Most computers should be able to handle 10k observations.

The dependencies should all be installable with pip.

To run the program, run main.py : "python main.py -w" . The "-w" option opens the software in windowed mode.

Some keys (all the other details are in the pdf):

"x" : main screen. Can be remapped in the config file.
After choosing a dataset, right click on the empty space to choose a DR algorithm and its hyperparameters.
Clicking in an embedding will select points.
Ctrl+click+keep pressed : select points by drawing a shape in the embedding.
Ctrl+hover without clicking : shows the high-dimensional values of the nearest point as a heatmap.
With some points selected: press "o" to open the selection in a new tab : the selection is now considered as a new dataset.
Right click on a dataset tab to close it.

"c" : relative QA sreen. Can be remapped in the config file.
Allows visual comparisons of the local and global quality of embeddings.

"v" : absolute QA sreen. Can be remapped in the config file.
Allows a visual assessment of the local and global quality of embeddings.
