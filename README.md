# ViolenceNet-Darpli

Three boys and a violent dream.


# File Structure

ViolenceNet-Darpli/

<pre> ```
├── code/
    ├── 2DCNN/
      ├── content_moderation.py
      ├── hyperparameters.py
      ├── live_detection.py
      ├── preprocess,py
      ├── 2D CNN WEIGHTS HERE
      ├── 
    ├── 3DCNN/
        ├── 3Dcnn_content_moderation/
            ├── Input_videos/
                ├── Videos...
            ├── Output_videos/
                ├── Videos...
            ├── 3Dcnn_vid_content_mod.py
            ├── 3D CNN WEIGHTS HERE
        ├── 3Dcnn_live_detection/
            ├── predict_live.py
            ├── 3D CNN WEIGHTS HERE
    ├── checkpoints/
        ├── SAVED MODEL INFORMATION (IF RUNNING TRAINING PIPELINE)
    ├── logs/
        ├── SAVED MODEL INFORMATION (IF RUNNING TRAINING PIPELINE)
    ├── misclassified/
    ├── weights-info
        ├── 2d-cnn-from-scratch.txt
        ├── 2d-cnn-resnet.txt
        ├── 3d-cnn.txt
    ├── 2D_testing.py
    ├── hyperparameters.py
    ├── main.py
    ├── models.py
    ├── preprocess.py
    ├── tensorboard_utils.py
├── data/
    ├── test
        ├── NV
            ├── frames...
        ├── V
            ├── frames...
        ├── _classes.csv
    ├── train
        ├── NV
            ├── frames...
        ├── V
            ├── frames...
        ├── _classes.csv
    ├── valid
        ├── NV
            ├── frames...
        ├── V
            ├── frames...
        ├── _classes.csv
    ├── README.dataset.txt
    ├── README/roboflow.txt
├── yolo
    ├── train
    ├── train4
├── .gitignore
├── README.md
├── reformat_data.py

        
``` </pre>

# Instructions

2D CNNs:

To run content moderation or live detection on our 2D CNN models, download the relevant weights files (your.e016-acc0.9152.weights.h5 and your.e045-acc0.9332.weights.h5) in this Google Drive folder: INSERT LINK

Place the weights files in the relevant directory: /code/2DCNN

If you want to run content_moderation.py on a video, you will need to edit the video_path variable on line 42. Additionally,for both content_moderation.py and live_detection.py the code is defaulted to be run with our from scratch 2D CNN (as opposed to our ResNet-based model) which has the weights 'your.e016-acc0.9152.weights.h5' but it can be changed to whatever weights you choose to use.

PROVIDE RESNET ARC IN COMMENTS IN CONTENT_MODERATION.PY??

Lastly, make sure to run the content_moderation.py and live_detection.py scripts from the /code/2DCNN directory.

3D CNNs:

To run content moderation or live detection on our 3D CNN model, download the relevant weights file (model_3dcnn_global_.94.h5) in this Google Drive folder: INSERT LINK

Place the weights files in the relevant directory: /code/3DCNN/3Dcnn_content_moderation or /code/3DCNN/3Dcnn_live_detection

For content moderation you have multiple options on whether to run predictions over every frame of a video or make a prediction over the entire video. You can also run predictions over multiple videos. To use the different options, edit the code at the bottom of 3Dcnn_vid_content_mod.py accordingly.

Lastly, make sure to run the 3Dcnn_vid_content_mod.py script from the /code/3DCNN/3Dcnn_content_moderation directory and the predict_live.py script from the /code/3DCNN/3Dcnn_live_detection directory. 

# Other Notes

The data in the data directory is the image data used to train our 2D CNN. The dataset used to train our 3D CNN can be found here: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset/data

We trained our 2D CNN models with the pipeline in this repository using remote computing resources.

We trained our 3D CNN with this Google Colab Notebook: https://colab.research.google.com/drive/1n63T816Q0pUBHftbJZlTxpvLemJjN0_u#scrollTo=0MfR0tOPxlZl


# Acknowledgements

We employ the training pipeline from Brown University's CSCI 1430 Homewokr 5 Assignment for the 2D CNNs. We employ the training and evaluation pipeline from Khalid's Google Colab Notebook: https://www.kaggle.com/code/pranav1718/real-time-violence-detection-mobilenet-bi-lstm
