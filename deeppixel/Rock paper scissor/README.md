# Rock Paper Scissor Game
A Rock-Paper-Scissors game using computer vision and deep learning.
where rock wins over paper , scissor wins over paper and paper wins over rock.

This repository contains ```play_with_computer``` file to play it live.

Preview:
![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Rock%20Paper%20Scissor/output/play_game.png?raw=true)
<br><br>

![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Rock%20Paper%20Scissor/output/play.gif?raw=true)

## Steps:
* Tarin the model using an appropriate dataset. Here,a mixed data set is used to provide variation to the model and to avoid over fitting .
I have created my own data set using python script and mixed it with the dataset available online. Here is the preview of images I have used.
* after training and testing the model ,save the model weights 
* Add model weights to the python script provided ```play_with_computer```.
* Run the file ensure that you have installed all the dependencies mentioned in ```requirements.txt```.

## Dataset Used :

### Rock 
![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Rock%20Paper%20Scissor/Input%20used/rock.png) ![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Rock%20Paper%20Scissor/Input%20used/rock_1.png?raw=true)
### Paper
![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Rock%20Paper%20Scissor/Input%20used/paper.png) ![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Rock%20Paper%20Scissor/Input%20used/paper_1.png) 
### Scissor
![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Rock%20Paper%20Scissor/Input%20used/scissor.png)   ![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Rock%20Paper%20Scissor/Input%20used/scissor_1.png) 

## Dependencies:
* tensorflow
* keras
* python3
* opencv

