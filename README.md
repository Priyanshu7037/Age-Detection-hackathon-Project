# Age-detection


# Abstract
- Combining people tracking with age detection is a good idea for many and many applications in real life scenarios such as store management to gather the information of customers for further analysis, or in/out people control for security purposes in buildings ...
- This is just a small step of putting the state-of-the-art image processing techniques together.
- # Requirements
- Install Python and neccessary libraries as Mentioned in project report 
# The project repository is organized as follows:
- face_age - To be downloaded from https://drive.google.com/drive/folders/1E9m9dZYLga9kc9NGPHfZa75JNJgpgv3M
  and to be saved in working Folder With name 'face_age'. 
- model3.0.jpynb - # Jupyter notebook for data exploration, model development, and 
evaluation
- 'my_model1' and 'saved_model' - To load save model (Steps to load saved model are given below)
- Readme file
# model3.0.jpynb -
 This is the joupyter file that contains our code for our Model training .
 For further details go through the project report.

# Requirements to load saved model
- Install Python
```
  pip install tendorflow
  pip install opencv
```
# Method to run saved_model
- Create a jupitor file 
- Befor executing model save 'my_model1' , 'saved_model' and 'model3.0' in working directory.
- To load saved model run -
```
import tensorflow as tf
loaded_model = tf.keras.models.load_model("saved_model")
```
- Save the testing image in working directory. 
- To predict the age of Saved image -
```
import cv2
import numpy as np

image = cv2.imread('Sample image for Real time testing.jpg') 

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

resized_image = cv2.resize(img_gray, (100,100))
image_np = np.array(resized_image)

resized_image = np.reshape(image_np, (-1, 100, 100, 1))

img_normalized = resized_image/ 255.0

prediction=loaded_model.predict(img_normalized)
max_position = np.argmax(prediction)

print("YOUR AGE IS: ")

if(max_position==1):
    print("0-5")
elif(max_position==2):
    print("6-12")
elif(max_position==3):
    print("13-18")
elif(max_position==4):
    print("19-30")
elif(max_position==5):
    print("31-45")
elif(max_position==6):
    print("46-65")
elif(max_position==7):
    print("66-80")
else:
    print(">81")

```

- For further details go through the project report.




# Result
- Accuracy for train and test data are 93.48 % and 60.93 % respectively.
- Predict age group of given image.




- For further details go through the project report.
# Acknowledgment
- Thanks to the open-source community for providing various machine learning 
libraries and tools.