# Delhi_NCR-Landcover-Prediction-using-CNN-AI-for-Sustainability-
## Delhi-NCR Landcover Prediction using a simple CNN model

All resources(shapefiles, sentinel-2 image file and the landcover.tif files) were taken from [here](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?)

The notebook datasetlabelingandtraintestvis.ipynb greatly deals with maneuvering through the shapefiles and the image files. The Delhi_ncr geojson shapefile plays a crucial role defining a fixed boundary which would aid in filtering out the sentinel-2 images. 

The delhi_ncr shapefile has been visualized using matplotlib in the above notebook and then a 60x60 grid overlay has been added for improving the perceptibility.


Sentinel-2 RGB image patches (128×128 pixels, 10m/pixel resolution), with each image filename has been mapped to its center coordinates.

```
folder="/home/aderham/delairshed/rgb/"
data=[]
for sentim in os.listdir(folder):
    if sentim.endswith(".png"):
        lat,lon=sentim.replace(".png","").split("_")
        data.append({
            "filename": sentim,
            "lat":float(lat),
            "lon":float(lon)
        })
df=pd.DataFrame(data)

gdfpts=gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"
)
```

This is then further used to filter satellite images whose center coordinates fall inside the region.
```
merged=merged.dissolve()
filteredmerge=gpd.sjoin(gdfpts,merged,predicate="within")
```

The total number of images reported before (9,216) and after (8,015) filtering has beeing observed in the above notebook.

For each image, the  128×128 corresponding land-cover patch from land_cover.tif is extracted using its center coordinate from the sentinal-2 image's name that contains the latitudes and longitudes.
```
patchsz=128
half=patchsz//2
patches=[]

for _, row in filtered.iterrows():
    lon,lat=row["lon"],row["lat"]

    x,y=transformer.transform(lon,lat)
    rowpx,colpx=src.index(x,y)

    win=Window(colpx - half, rowpx - half, patchsz, patchsz)
    patch=src.read(1,window=win)

    patches.append(patch)
    
```

The image label has been assigned using the dominant (mode) land-cover class. 
```
labels = []
for patch in patches:
    values,counts=np.unique(patch,return_counts=True)
    domclass=values[np.argmax(counts)]
    labels.append(domclass)
```
ESA class codes have been mapped to simplified land-use categories (e.g., Built-up, Vegetation, Water, Cropland, Others). 
```
esa_to_label = {
    10: "Tree_cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare_sparse_veg",
    70: "Snow_ice",
    80: "Water",
    90: "Herbaceous_wetland",
    95: "Mangroves",
    100: "Other",
}
```

Perform a 60/40 train-test split randomly and visualize class distribution 
```
train,test=train_test_split(filtered, test_size=0.4, random_state=42)
print("Train:\n",train_df["lclabel"].value_counts())
print("\nTest:\n",test_df["lclabel"].value_counts())
```
The class distributiob has been visualized in the notebook with bar-graphs which show and uneven and irregular data distribution as the dataset has been divided randomly.
```
train_counts = train_df["lclabel"].value_counts()
test_counts = test_df["lclabel"].value_counts()

dfplt = pd.DataFrame({"Train": train_counts,"Test": test_counts})
dfplt.plot(kind="bar")

plt.title("Class Distribution: Train vs Test")
plt.ylabel("Count")
plt.xlabel("Class")
plt.xticks(rotation=45)
plt.show()
```

A simple CNN model was implemented (CNNimplementation.ipynb) using the pytorch "nn" framework which has the convolutional layers, pooling layers and fully connected layer. The fully-connected layer uses a ReLu activation function which aids in effectively managing the loss of the model by dealing with the vanishing-gradient problem.

Adam optimizer has been used to effectively scale the learning rate by keeping track of the moving average of squared gradients.

The data loader is used for creating batch size of 32 and shuffle the training set so that the model doen't get used to the datapoints' order. shuffle = False for testing to maintain a steady evaluation sequence.

Accuracy and F1 score, precision, recall and error rate has been evaluted using the test set which shows that the model is overall generallizing well. 
```
Test Accuracy: 80.29%
Error Rate: 0.1971

Confusion Matrix:

[[ 318  314    0   16   34    0]
 [   4 2197    0    4   12    0]
 [   3   54    0    0    0    0]
 [   2   94    0   10    4    0]
 [   8   77    1    1   49    0]
 [   0    3    0    0    1    0]]

Classification Report:

              precision    recall  f1-score   support

           0       0.95      0.47      0.63       682
           1       0.80      0.99      0.89      2217
           2       0.00      0.00      0.00        57
           4       0.32      0.09      0.14       110
           5       0.49      0.36      0.42       136
           6       0.00      0.00      0.00         4

    accuracy                           0.80      3206
   macro avg       0.43      0.32      0.34      3206
weighted avg       0.79      0.80      0.77      3206

```

## Inference:

### The CNN model used with the Adam optimizer and on 5 epochs of training with a batch-size of 32 has yeilded reasonable performance.
### The training accuracy states 86.40% in the last epoch while the testing accuracy showcases 80.29% which shows slight overfitting. This can be taken care of by introducing regularization with respect to dropouts.

### A stratified approach towards the train-test split would also have a great deal in preventing the class imbalance issue the model is currently facing; the zero_division parameter warning.

### Slightly adjusting the learning rate and increasing epochs would also greatly aid in increasing the model performance.

### Data augmentation or deeper CNN can aid in improving the feature map created by the simple CNN.

### Using other varaints like Alexnet or ResNet18 may greatly improve the performance.

### A rigorous validation evaluation would aid a great deal in thoroughly tuning the hyperparameters.

### Increasing the training data would also aid the model in improving the overall performance.



## Acknowledgement
### Resources used
[Pytorch Documentation for the "nn" framework](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

[Evaluatio Metrics](https://www.datasciencesmachinelearning.com/2018/11/confusion-matrix-accuracy-precision.html)

[Working with geopandas for using .geojson files](python-graph-gallery.com/map-read-geojson-with-python-geopandas/)

[Using Rasterio for ".tif"](https://rasterio.readthedocs.io/en/stable/quickstart.html)

#### Usage of AI
AI has been used for basic logic relays, for understanding the concept. It has been used for properly learning and understanding the key concepts behind CNN and deep neural networks. It has also been used for validating or cross-verifying my approaches.
