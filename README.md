# MDTS-Net
An multiscale GAN for dopamine transporter single-photon emission computed tomography (DAT-SPECT) simulation from MRI images.

Reference:
- https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging

*******************************************************************************
## Functions of each scripts

- organize_folder_structure.py: Organize the data in the folder structure (training,testing) for the network, and align the coordinates of images and labels.

- check_loader_patches: Shows example of patches fed to the network during the training.

- options_folder/base_options.py: List of base_options used to train/test the network.  

- options_folder/train_options.py: List of specific options used to train the network.

- options_folder/test_options.py: List of options used to test the network.

- utils_folder: contains the Nifti_Dataset Dataloader and augmentation functions to read and augment the data.

- models_folder: the folder contains the scripts with the networks and the cycle-gan training architecture.

- train.py: Runs the training. (Set the base/train options first)

- test.py: It launches the inference on a single input image chosen by the user. (Set the base/train options first)

- result.py: Calculates the PSNR, SSIM, FID of each output image.
*******************************************************************************
## Usage
### Folders structure:

Use first "organize_folder_structure.py" to create organize the data.
Modify the input parameters to select the two folders: images and labels folders with the dataset.

    .
	├── Data_folder                   
	|   ├── MRI               
	|   |   ├── image1.nii 
    |   |   ├── image2.nii 	
	|   |   └── image3.nii                     
	|   ├── DAT-SPECT                        
	|   |   ├── image1.nii 
    |   |   ├── image2.nii 	
	|   |   └── image3.nii  

Data structure after running it:

	.
	├── Data_folder                   
	|   ├── train              
	|   |   ├── images (MRI)            
	|   |   |   ├── 0.nii              
	|   |   |   └── 1.nii                     
	|   |   └── labels (DAT-SPECT)            
	|   |   |   ├── 0.nii             
	|   |   |   └── 1.nii
	|   ├── test              
	|   |   ├── images (MRI)           
	|   |   |   ├── 0.nii              
	|   |   |   └── 1.nii                     
	|   |   └── labels (DAT-SPECT)            
	|   |   |   ├── 0.nii             
	|   |   |   └── 1.nii
	


