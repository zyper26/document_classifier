# document_classifier

### How I created the dataset

1. Manual downloaded all file as scraping was not possible on the website and it required authentication.
2. Looked into the data and identified that 1st page is enough to identify which form type is it.
3. Converted the pdfs 1st page to images.
4. Labelled each image using [Robot VGG tool](https://www.robots.ox.ac.uk/~vgg/software/via/via.html). Code file for this used is ```data_labeling.ipynb```


### Dataset 




### Evaluation Metrics

1. Here we used accuracy as the metrics. However, better approach would have been to use f1-score rather than accuracy as we have imbalanced dataset.

### Future Works

1. Add more data to get better accuracy.
2. Here we have only worked on image classification (reason being all the form are structured). We can also try to extract the text using tesseract and use text classification on it but due to lack of data I think we will not able to achieve good accuracy. 
   ```
   import pytesseract
   ocrd_text = pytesseract.image_to_string(Image.open('image.jpg'))
   ```

   
