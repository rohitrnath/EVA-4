# Assignment 5

##STEP 1 - Step1BaseModel.ipynb

###Target:
* Setup the Base model
* Set transforms
* Set Data loader
* Set Basic skelton code ready
* Set basic training and testing loop

###Results:
* Parameters 				-	194,884
* Best Training Accuracy 	-	99.38%
* Best Testing Accuracy		-	98.94%

###Analysis:
* Heavy model for MNIST problem
* Model is over-fitting


##STEP 2 - Step2LightModelWithBN-GAP.ipynb

###Target:
* Made the model lighter to achieve less number of parameters
* BatchNormalisation added
* GlobalAveragePooling added as second last layer

###Results:
* Parameters 				-	7,432
* Best Training Accuracy 	-	99.65%
* Best Testing Accuracy		-	99.28%

###Analysis:
* Drastic improvement in accuracies after adding BatchNorm
* GAP layer reduced model capacity(Number of Params)
* Model is still over-fitting


##STEP 3 - Step3Regularize.ipynb

###Target:
* Added dropout to all layers
* Tested with different dropout values

###Results:
* Parameters 				-	7,432
* Best Training Accuracy 	-	99.26%
* Best Testing Accuracy		-	99.29%

###Analysis:
* Dropout reducing the gap between testing and training accuracy.
* Dropout reducing the accuracy.


##STEP 4 - Step4Augmentation.ipynb

###Target:
* Image augmentation added to dataset with random rotation of -10 to 10 degree

###Results:
* Parameters 				-	7,432
* Best Training Accuracy 	-	99.00%
* Best Testing Accuracy		-	99.34%

###Analysis:
* Image augmentation results far better testing accuracy than training


##STEP 5 - Step5-BS-stepLR.ipynb

###Target:
* Introduced step learning rate.
* Batch size changed to 64.

###Results:
* Parameters 				-	7,432
* Best Training Accuracy 	-	98.98%
* Best Testing Accuracy		-	99.43%

###Analysis:
* Got steady increase in test accuracy and 99.43% of test accuracy got at 20th epoch.
* Step learning rate helps to reduce oscilation and improved training at higher epochs.
* Seems like batch size also influncing the training accuracies.