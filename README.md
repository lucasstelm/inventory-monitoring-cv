# Inventory Monitoring at Distribution Centers

Distribution centers often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects.

The objective of this project is to develop a model capable of counting the number of objects within each bin. Such a system can be utilized for inventory tracking and ensuring accurate item quantities in delivery consignments.

To construct this project, AWS SageMaker will be used to retrieve data from a database, preprocess it, and train a machine learning model. Additionally, an endpoint will be created using SageMaker to deploy the trained model. This endpoint can serve as an interface for making predictions on new data, allowing the model to be integrated into the distribution center's operations.

## Dataset

### Overview

The Amazon Bin Image Dataset contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. Each image is accompanied by a metadata file containing information such as the object count, dimensions, and object types.

The  data  is  distributed  under  the  Creative  Commons  Attribution-NonCommercial-ShareAlike 3.0 and can be retrieved from the public S3 bucket named **aft-vbi-pds**. More information about the data is available at this [link](https://registry.opendata.aws/amazon-bin-imagery/).

Initially,  a  restricted  subset  of  images  will  be  utilized  to  test  the  feasibility  of  the idea  and  validate  the  machine  learning  pipeline.  This  subset  will  serve  as  a  proof  of concept to ensure that the pipeline functions as intended. Once the initial testing phase is completed and the machine learning pipeline is deemed successful, the model can then be retrained with additional data.

## Model Training

The project requires a robust solution for multi-class classification of images.  The  model  chosen for this project was  Convolutional Neural  Networks  (CNNs),  as  they  have  revolutionized  the field  of  computer  vision  by significantly improving the accuracy and efficiency of image analysis tasks. One  especially  important  feature  of  this  approach  is  the  ability  to  use  transfer learning, which involves using pre-trained models trained on large datasets like ImageNet. These pre-trained models have learned general features from a vast amount of data and can be fine-tuned or used  as a feature  extractor  for  specific  image classification tasks with limited training  data.  Transfer learning  allows  for  faster  convergence  and  improved performance, especially when labeled training data is scarce.

The hyperparameters learning and batch size were tuned, since they are crucial in training Convolutional Neural Networks (CNNs). The learning rate determines the step size for weight updates during training, impacting convergence speed, stability, and generalization. The batch size affects computational efficiency, generalization, memory usage, and stability by determining the number of samples processed before weight updates. Finding the optimal values through hyperparameter tuning is necessary to ensure efficient and effective training of CNN models.

After tuning the hyperparameters, the model was retrained with the best values and for 20 epochs. The results appear below.

| Class   | Precision | Recall | F1     |
| ------- | --------- | ------ | ------ |
| Class 1 | 0.5882    | 0.4918 | 0.5357 |
| Class 2 | 0.3525    | 0.3772 | 0.3644 |
| Class 3 | 0.3050    | 0.4586 | 0.3664 |
| Class 4 | 0.3982    | 0.3814 | 0.3896 |
| Class 5 | 0.2121    | 0.0753 | 0.1111 |

| Average Test Accuracy | 0.3584 |
| --------------------- | ------ |
| Average Test Loss     | 1.4073 |

## Machine Learning Pipeline

In this project, we have successfully developed a machine learning workflow leveraging the capabilities of AWS (Amazon Web Services) services. Our primary objective was to train and deploy an image classification model using these services, and we have achieved that goal.

The initial step involved obtaining image data of products inside bins from an Amazon bucket. To facilitate further analysis, each image was labeled by moving it to a specific folder corresponding to its class, which was determined based on the number of objects present in the image. The labeled images were then uploaded to a private S3 bucket, which served as a storage and data source for subsequent steps.

To optimize the performance of the Convolutional Neural Network (CNN) model, hyperparameter tuning was conducted, with a focus on two key parameters: the learning rate and batch size. PyTorch, a popular deep learning framework, was utilized for this purpose. Different combinations of learning rates and batch sizes were tested, and the model's performance was evaluated using suitable metrics to identify the best hyperparameters.

Once the optimal values for the learning rate and batch size were determined, the model was retrained using the labeled images and the selected hyperparameters. The training process involved feeding the data through the CNN, adjusting the model's weights and biases based on the calculated loss, and iteratively updating the parameters to minimize the error. This retraining phase aimed to improve the model's accuracy and predictive capabilities.

After the model had been retrained, it was ready for deployment. This involved making the model accessible for queries or predictions. The deployed model was tested by querying it with new or unseen images to assess its performance and verify its ability to accurately classify the number of objects present in the product images. The results of these queries provided insights into the model's effectiveness and helped evaluate its suitability for the intended application.

To further improve the model with more computing power, we can increase the number of training epochs to allow for better parameter fine-tuning and capturing complex patterns. Utilizing high-performance GPUs can expedite the training process and potentially enhance performance. Distributed training across multiple instances can accelerate convergence and yield improved results. Exploring more sophisticated model architectures, conducting extensive hyperparameter tuning, and leveraging ensemble learning techniques can also contribute to enhancing the model's performance. However, it's important to consider the associated costs and resource requirements when allocating additional computing power.

By utilizing AWS services, we established an end-to-end machine learning workflow that encompassed data storage, compute resources, model training, and monitoring. The integration of these services enabled us to efficiently train an initial model, bringing us closer to achieving our project objectives.
