# BehaviorDetection

##Before running
First, you need to install DeepLabCut (https://github.com/DeepLabCut/DeepLabCut) and train your own DeepLabCut key points detection model. Detecting your video, obtain key points detection files (move to analyze folder) and corresponding metadata files (move to metadata folder). The labeled key points name in this project are called front and posterior. In the later step, the input key points name should correspond to the label. 

Grooming behaviors detectable in this project and corresponding label names: {'0': 0,' head': 1,' forefoot': 2,' forefoot': 3,' hindfoot': 4,' hindfoot': 5,' abdomen': 6,' wings': 7}

##Installation
'pip install -r requirements.txt'

##How to use?
We provide the corresponding data file 'format (.csv)' to format and save the label file. Please enter your own label data before running.

* Label data extraction (record by human vision): 
Running 'random_create_data.py' which will read the label data in '1_label_by_human.csv', randomly generate the extracted frame, and save the file as '2_random.csv'.

* Training data set generation: 
Running 'exract_training_data.py', it requires set the path of the '2_random.csv', 'detect folder', 'video folder' and 'storage folder' (refer to 'training_data_base folder' for format). After running, the label file and images data will be generated, and the images data will be stored in the 'label folder'. 

* Training: 
Divide the training data into Train and val (optional) in proportion, and keep the structure unchanged. Run 'train_one_network.py' to set the 'folder path' of train and val data, and 'debug other parameters' according to your needs. 

* Model evaluation: Run 'evaluate.py'.

* Recognition and count the grooming behavior in video: 
First, train your model and detect the key points information of the video. Run 'DetectVideo.py' to realize end-to-end grooming behavior statistics, and pay attention to changing the path.
