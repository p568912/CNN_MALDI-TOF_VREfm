# maldi-tof
In this project, we develop the deep learning algorithm to classify bacteria with vanocomycin resistance (VREfm) or not according to the MS data. Our performance is superior than previous methods such as tradition machine learning. In addition, our code can tell users which m/z range is more important for any samples. We will give score for each m/z position by normalizing from 0 to 1 in this sample. If the score is greater,it will represent the m/z position more important based on the model. We hope our algorithm can be applied to resistance classifcation of any kind of bacteria.

## workflow
We divde into three sections.

### preprocessing: (preprocess.py)
MS data is processed and it will be transformed as deep learning model input after preprocessing.

### training: (dataset.py ,)
In this section, we will train the model from our code using preprocessed input in previous step. Users can modify parameters to change the architeture of network based on demands. (ex:poolingFlag,ReLUFlag,channels,batch_size, etc.)

### model interpretation: (scorecam.py)
We will


