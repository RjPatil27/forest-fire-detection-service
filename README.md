# forest-fire-detection-service
Service based approach for early-stage fire and smoke detection with the help of Convolutional Neural Network (CNN)

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run start
```

### Compiles and minifies for production
```
npm run build
```

### Lints and fixes files
```
npm run lint
```

### Service Hypothesis
For the demonstration purpose, we have assumed that the vision sensors are deployed in the forest area on the upper canopy level and lower. We have also considered that some vision sensors are deployed on high vantage points such as watchtowers, mobile cell towers, and transmission towers. These vision sensors will send periodically captured images or continuously send video streams to the backend service. To demonstrate this scenario, we have created a user interface that will allow the end-user to upload an image or video. If the service detects fire or smoke in a real-life scenario, an alert will be sent to the respective authority.

### Dataset
Our dataset is comprised of around 6000 fire and non-fire images collected from Github, Google Drive, and Kaggle. Subsets of that dataset will be used to train, validate, and test the Inception V3. We tried our best to the images related to the forest and forest fires. We also have a video dataset of 15 videos. Out of which, four videos were used for training and validation of the ResNet50 convolutional neural network.

### Architecture
The web service has two main parts:
* Flask Backend Application
* User Interface

The user interface is built for the demonstration of the service hypothesis scenario.
![Architecture](https://github.com/rohitmokashi16/forest-fire-detection-service/blob/main/images/Architecture.png)

###
