Experimentation Log 

Intial setup was as follows:

One convolutional layer, 32 filters, kernel size (3,3)
One max-pooling layer, pool_size=(2,2)
One hidden layer, 128 neurons
Intial accuracy = 0.34

->After trying out different number of hidden layers, it was observed atleast three hidden layer were required to achieve accuracy greater than 0.90.

->After adding another layer of convolution and pooling, accuracy increased to 0.90 with three hidden layers. Adding another layer led to further improvement.

->Further more, when number of filters were used in ascending order such as (32, 64, 128), both time and accuracy improved.

->Intially using smaller filter to capture local information and using larger filter on subsequent convolutional levels can bring increase in accuracy.

->When convolutional layers were optimized as above, differences in accuracy level was insignificant with respect to number of hidden layer. Accuracy level remained between 0.96 and 0.97
  when hidden layers were either 1,2 or 3 in number.

-> Dropout between 0.2 and 0.3 seemed to work the best.

 
