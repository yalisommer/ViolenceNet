# ViolenceNet-Darpli: Harper Austin, Yali Sommer, Daniel Schiffman

With this project, we explore the use of convolutional neural networks (CNNs) to solve abstract classification problems. Specifically, we tackle the problem of violence detection (in the context of person-to-person interactions such as street fights). Despite being a binary task (Violent or NonViolent), it is a nuanced task with noisy data, and the differences between ordinary, non-violent physical interactions and violent ones can at times be quite subtle, e.g. a hug and a wrestling attack can look quite similar. 

We first attempt to solve the problem using a 2D CNN trained on frames of violent and non-violent videos. We train a 2D CNN from scratch and we finetune a CNN with ResNet features as a backbone. We then train a 3D CNN from scratch on violent and non-violent videos.

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

The dataset used to train our 2D CNNs can be found here: https://universe.roboflow.com/dinesh-nariani-rmnpr/violence-not_violence-ziv7b/dataset/2 . The reformat_data.py script can be used to reformat this dataset in order to use it with the 2D CNN training pipeline.

The dataset used to train our 3D CNN can be found here: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset/data

We trained our 2D CNN models with the pipeline in this repository using remote computing resources.

We trained our 3D CNN with this Google Colab Notebook: https://colab.research.google.com/drive/1n63T816Q0pUBHftbJZlTxpvLemJjN0_u#scrollTo=0MfR0tOPxlZl



# Acknowledgements

We employ the training pipeline from Brown University's CSCI 1430 Homewokr 5 Assignment for the 2D CNNs. We employ the training and evaluation pipeline from Khalid's Google Colab Notebook: https://www.kaggle.com/code/pranav1718/real-time-violence-detection-mobilenet-bi-lstm
