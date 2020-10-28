#--------------------------------------------#
 MNIST Hand Written Digits Dataset problem
 
 This is a deep neural network that can be
 trained to recognize the handwritten digits 
 from the MNIST dataset 
 
 Tncluded is a pickel that can be imported 
 and will recognize the testing data with
 a 94.45% accuracy. It is a nn with 2
 hidden layers of 16 neurons that was trianed 
 for 3000 epochs. During this training it 
 peaked at 95.4% accuracy at ~1700 epochs
 

 It still needs the following:
 - fix drawing nn for user input
   - the nn has a very low accuracy 
     on user input data -likely
     because the nn relies on the 
     low values to recognize edges

 Version: 10-23-2020
#--------------------------------------------#


#--------------------------------------------#
 Also github wont let me upload the data 
 so you gotta go to 
 http://yann.lecun.com/exdb/mnist/
 and download the 4 files unzip them and 
 rename them so that the dot after 'images' 
 or 'labels' becomes a dash

 just put the data in a folder and update the 
 directory in 'mnistload.py'

 you also need to install a few python modules

 - numpy 
 - mnist
 - pickel
#--------------------------------------------#
