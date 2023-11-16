# document_classifier

### How I created the dataset

1. Manual downloaded all file as scraping was not possible on the website and it required authentication.
2. Looked into the data and identified that 1st page is enough to identify which form type is it.
3. Converted the pdfs 1st page to images.
4. Labelled each image using [Robot VGG tool](https://www.robots.ox.ac.uk/~vgg/software/via/via.html). Code file for this used is ```data_labeling.ipynb```

### Train

1. Keep the data in labelled_data folder. I have demonstrated 3 classes only but we have 9 classes overall in labelled_data
  - **labelled_data:**
     - `Form_D/`
     - `Form_6-K/`
     - `Other/`
2. Run ```main.py```. Here I have not created the environment file yet so you can use pytorch environment to train the model. 

### Dataset 

![Dataset](https://github.com/zyper26/document_classifier/blob/main/dataset_classes.png)


### Evaluation Metrics

1. Here we used accuracy as the metrics. However, better approach would have been to use f1-score rather than accuracy as we have imbalanced dataset. Below you can see the accuracy (only represented with tqdm bar):

   ![Train data valuation](https://github.com/zyper26/document_classifier/blob/main/model_svaing.png)



### Future Works

1. Add more data to get better accuracy.
2. Here we have only worked on image classification (reason being all the form are structured). We can also try to extract the text using tesseract and use text classification on it but due to lack of data I think we will not able to achieve good accuracy. 
   ```
   import pytesseract
   ocrd_text = pytesseract.image_to_string(Image.open('image.jpg'))
   ```
3. Instead of cross entropy loss we should use focal loss.
4. Lot of experimentation needs to be done to check which model would give us better accuracy.
