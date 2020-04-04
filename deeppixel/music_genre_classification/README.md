# Music Genre Classification

## Aim :
Audio data is gaining a lot of importance in machine learning world. Procsessing, Generation and Classification of music audios is gaining a lot of attention. In this repository, we aim to predict the genre of the input audio.

## Dataset used :
We are using GTZAN dataset. It can be downloaded from http://marsyas.info/downloads/datasets.html. This dataset is contains 10 genres. With 100 songs per genre, we have a total of 1000 songs. The genres present in this dataset are :

1.blues

2.classical

3.country

4.disco

5.hiphop

6.jazz

7.metal

8.pop

9.reggae

10.rock


## Approach :

In this approach, we convert our audio signals into spectograms. These spectograms are in the form of images.
Just like, we humans have a unique thumbprint, we can say that, each genre of music generates a spectogram. 
For every genre, the spectograms generated have some unique features, which differentiates it from other spectograms. 
Thus, we treat this time vs frequency generation (spectogram) as an image, which is fed into our CNN model. We have made a custom made CNN model, which takes input of shape (720,720,4). This is same as the shape of the spectogram generated.

Order of display of example spectograms (Blues,Classical,Country,Disco,Hiphop,Jazz,Metal,Pop,Reggae,Rock)

<img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/blues_genre_spectrogram.png"  title="Blues"/><img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/classical_genre_spectrogram.png" title="Classical"/><img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/country_genre_spectrogram.png" title="Country"/><img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/disco_genre_spectrogram.png" title="Disco"/><img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/hiphop_genre_spectrogram.png" title="Hiphop"/><img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/jazz_genre_spectrogram.png" title="Jazz"/><img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/metal_genre_spectrogram.png" title="Metal"/><img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/pop_genre_spectrogram.png" title="Pop"/><img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/reggae_genre_spectrogram.png" title="Reggae"/><img src="https://github.com/purva98/DeepPixel/blob/master/deeppixel/music_genre_classification/spectogram/rock_genre_spectrogram.png" title="Rock"/>
