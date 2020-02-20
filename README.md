# DeepStyle
### Background: 🔍

Deep Neural networks have already surpassed human level performance in tasks such as object recognition and detection. However, deep networks were lagging far behind in tasks like generating artistic artefacts having high perceptual quality until recent times. Creating better quality art using machine learning techniques is imperative for reaching human-like capabilities, as well as opens up a new spectrum of possibilities. And with the advancement of computer hardware as well as the proliferation of deep learning, deep learning is right now being used to create art. For example, an [AI generated art](https://www.christies.com/features/A-collaboration-between-two-artists-one-human-one-a-machine-9332-1.aspx) won’t be sold at an auction for a whopping $432,500.

Neural style transfer is one of the most creative application of convolutional neural networks. By taking a content image and a style image, the neural network can recombine the content and the style image to effectively creating an artistic image!
These algorithms are extremely flexible and the virtually infinite possible combinations of content and style resulted in very creative and unique results.

### Bird Eye View:

Neural style transfer uses a pretrained convolution neural network. Then to define a loss function which blends two images seamlessly to create visually appealing art, NST defines the following inputs:
* A content image (c) — the image we want to transfer a style to
* A style image (s) — the image we want to transfer the style from
* An input (generated) image (g) — the image that contains the final result (the only trainable variable)
