# ds-ga-1008-a2

K means pre-classification 

need to first replace the unsup.kmeans module with the one in this folder

training the kmeans
```
th train.lua -d extra -i 200
```

make prediction on the unlabel dataset(you may need to put the model file path as parameter as [-m model])
```
th classify.lua
```






