# ViolenceNet: Harper Austin, Yali Sommer, Daniel Schiffman

In this project, we explore the use of convolutional neural networks (CNNs) to solve abstract classification problems. Specifically, we tackle the problem of violence detection (in the context of hand-to-hand person-to-person interactions such as street fights). Despite being a binary task (Violent or NonViolent), it is a nuanced task with noisy data, and the differences between ordinary, non-violent physical interactions and violent ones can at times be quite subtle, e.g. a hug and a wrestling attack can look quite similar. 

We first attempt to solve the problem using a 2D CNN trained on frames of violent and non-violent videos. We train a 2D CNN from scratch and we finetune a CNN with ResNet features as a backbone. We then train a 3D CNN from scratch on violent and non-violent videos.

For both 2D and 3D CNNs we setup live video processing (from your computer's camera) and video processing to be able to make predictions on live video or prerecorded footage.

**Weights for our models and more information about the project can be found in this Google Drive folder:** https://drive.google.com/drive/folders/1LJYUbPcg2kAzXzUHZaHM7guwViDE5bEs

# File Structure

ViolenceNet/

<pre> 
├── code/
    ├── 2DCNN/
      ├── content_moderation.py
      ├── hyperparameters.py
      ├── live_detection.py
      ├── preprocess.py
      ├── 2D CNN WEIGHTS HERE 
    ├── 3DCNN/
        ├── 3Dcnn_content_moderation/
            ├── Input_videos/
                ├── YOUR VIDEOS HERE
            ├── Output_videos/
                ├── OUTPUT VIDEOS WILL GO HERE
            ├── 3Dcnn_vid_content_mod.py
            ├── 3D CNN WEIGHTS HERE
        ├── 3Dcnn_live_detection/
            ├── yolo_and_3d_live.py
            ├── 3D CNN WEIGHTS HERE
    ├── checkpoints/
        ├── SAVED MODEL INFORMATION (IF RUNNING 2D CNN TRAINING PIPELINE)
    ├── logs/
        ├── SAVED MODEL INFORMATION (IF RUNNING 2D CNN TRAINING PIPELINE)
    ├── misclassified/
        ├── SAVED MODEL INFORMATION (IF RUNNING 2D CNN TRAINING PIPELINE)
    ├── hyperparameters.py
    ├── main.py
    ├── models.py
    ├── preprocess.py
    ├── tensorboard_utils.py
├── PLACE data HERE
├── yolo
    ├── train/
    ├── train4/
    ├── yolo_detect.py
├── .gitignore
├── README.md
├── reformat_data.py
 </pre>

# Instructions

**2D CNNs:**

In order to train 2D CNN models, download the relevant dataset (referenced in Other Notes), rename the directory 'data' and place it in the root directory as indicated above. Then run the reformat.py script in order to format the data properly for training. In order to train the 2D CNNs, use the command line arguments provided for Task 1 in the CSCI 1430 Homework 5 Handout. 

To run content moderation or live detection on our 2D CNN models, download the relevant weights files (your.e016-acc0.9152.weights.h5 and your.e045-acc0.9332.weights.h5) in this Google Drive folder: https://drive.google.com/drive/folders/1LJYUbPcg2kAzXzUHZaHM7guwViDE5bEs

Place the weights files in the relevant directory: /code/2DCNN

If you want to run content_moderation.py on a video, you will need to edit the video_path variable. Additionally, for both content_moderation.py and live_detection.py the code is defaulted to be run with our from scratch 2D CNN (as opposed to our ResNet-based model) which has the weights 'your.e016-acc0.9152.weights.h5' but in comments is the way to run it for our ResNet-based model and it can be changed to whatever weights you choose to use.

Lastly, make sure to run the content_moderation.py and live_detection.py scripts from the /code/2DCNN directory.

**3D CNNs:**

To run content moderation or live detection on our 3D CNN model, download the relevant weights files (model_3dcnn_global_.94.h5 or model_3dcnn9394.h5) in this Google Drive folder: https://drive.google.com/drive/folders/1LJYUbPcg2kAzXzUHZaHM7guwViDE5bEs

Place the weights files in the relevant directory: /code/3DCNN/3Dcnn_content_moderation or /code/3DCNN/3Dcnn_live_detection

For content moderation you have options on whether to run predictions over every frame of a video with the predict_frames() method or make a prediction over the entire video by taking the majority of the model's predictions over every 16-frame chunk of the video with the predict_video() method. You can also run predictions over multiple videos. To use the different options, edit the code at the bottom of 3Dcnn_vid_content_mod.py accordingly.

In order to replicate our experiment on YouTube content moderation, you can download the videos included in the google drive link and put them in the Input_videos/ directory (make sure to only put the .mp4 files in and not inside any other folders). You can then run the 3Dcnn_vid_content_mod.py file as is to reproduce our results. This will take a while to run locally. Feel free to adjust any of the thresholds in this file or add your own data. If you do so, adjust the predict_violent_vids()
and predict_nonviolent_vids() methods accordingly.

The content moderation is defaulted to run with the model_3dcnn9394.h5 weights and the live detection is defaulted to run with the model_3dcnn_global_.94.h5 weights but these can be changed to use the other 3D CNN weights well.

Lastly, make sure to run the 3Dcnn_vid_content_mod.py script from the /code/3DCNN/3Dcnn_content_moderation directory and the yolo_and_3d_live.py script from the /code/3DCNN/3Dcnn_live_detection directory. 

# Other Notes

The dataset used to train our 2D CNNs can be found here: https://universe.roboflow.com/dinesh-nariani-rmnpr/violence-not_violence-ziv7b/dataset/2. The reformat_data.py script can be used to reformat this dataset in order to use it with the 2D CNN training pipeline.

The dataset used to train our 3D CNN can be found here: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset/data

We trained our 2D CNN models with the pipeline in this repository using remote computing resources.

We trained our 3D CNN with this Google Colab Notebook: https://colab.research.google.com/drive/1n63T816Q0pUBHftbJZlTxpvLemJjN0_u#scrollTo=0MfR0tOPxlZl



# Acknowledgements

We employ the training pipeline from Brown University's CSCI 1430 Homework 5 Assignment (https://browncsci1430.github.io/hw5_cnns/) for the 2D CNNs. We reference the training and evaluation pipeline from Khalid's Google Colab Notebook( https://www.kaggle.com/code/abduulrahmankhalid/real-time-violence-detection-mobilenet-bi-lstm) in our Notebook.
