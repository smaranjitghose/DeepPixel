# deeppixel 🐱‍💻 📷

![Issues](https://img.shields.io/github/issues/smaranjitghose/DeepPixel)
![Pull Requests](https://img.shields.io/github/issues-pr/smaranjitghose/DeepPixel)
![Forks](https://img.shields.io/github/forks/smaranjitghose/DeepPixel)
![Stars](https://img.shields.io/github/stars/smaranjitghose/DeepPixel)
[![License](https://img.shields.io/github/license/smaranjitghose/DeepPixel)](https://github.com/smaranjitghose/DeepPixel/blob/master/LICENSE)
[![Join the chat at https://gitter.im/opendeeppixel/community](https://badges.gitter.im/opendeeppixel/community.svg)](https://gitter.im/opendeeppixel/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

### Background: 🔍

Deep Neural networks have already surpassed human-level performance in tasks such as object recognition and detection. However, deep networks were lagging far behind in tasks like generating artistic artifacts having high perceptual quality until recent times. Creating better quality art using machine learning techniques is imperative for reaching human-like capabilities, as well as opens up a new spectrum of possibilities. And with the advancement of computer hardware as well as the proliferation of deep learning, deep learning is right now being used to create art. For example, an [AI-generated art](https://www.christies.com/features/A-collaboration-between-two-artists-one-human-one-a-machine-9332-1.aspx) won’t be sold at an auction for a whopping \$432,500.

### Our Vision: 🔍📃

Create a Python package📦 for plug in and play😎 different image processing and deep learning tasks without having to know about its working and the scary math that follows 😱.

In the process of doing so, we want the participants to:

- Learn various concepts in Deep Learning and Computer Vision🐱‍🏍
- Implement them in the form of scripts 👩‍💻
- Reimplement using more optimized methods and do a comparative analysis 🕵️‍♀️
- Create your own datasets, test your scripts on them and make it available for the community to work further 🔨
- Build small APIs for each task (wherever applicable)
- Build web apps for certain tasks using TensorFlow.JS or Flask
- Build Flutter apps with TensorFlow lite for certain tasks 📲
- Increment your personal portfolio with certain tasks you do here as standalone projects 👩‍💼
- Work for applying the various tasks undertaken into several real world scenarios
- Write blog posts about your contributions (Medium: @techquilla) to share with the world ✏
- Learn to read and implement research papers 🔬

## Guidelines and Suggestions to Contribute : 🤚 🏗

- **Don't push** anything to the root directory of the master branch. Always use specific subdirectories inside the [deeppixel](https://github.com/smaranjitghose/DeepPixel/tree/master/deeppixel) directory

For each of the tasks:

- Work on the task inside the `respective` sub-directory **[Please name it appropiately and use [camel_case](https://medium.com/better-programming/string-case-styles-camel-pascal-snake-and-kebab-case-981407998841)]**
- In the first attempt 💭💭 use a [Jupyter notebook] to perform your work.
- Once you are done give a Pull Request🩹 with the message 📩`Developed Jupyter Notebook for respective_task` , breifing about your approach in the description and add a link of the above notebook in [Google Colab](https://colab.research.google.com/) **[Please ensure you have given access]** ⛔
- Once merged😎 , build a script for the same task in the `respective` directory \_\_(If you are using Deep Learning, ensure that you have saved your trained model and its weights so that in the script you build can simply fetch it instead of training again)
- Use [argparse](https://docs.python.org/3/library/argparse.html) library so that the input image and output path can be given as arguments in the terminal while running the script
- Update the `requirements.txt` file in root directory of the master branch to ensure any additional modules you have used in present there.
- Make sure you provide sample images/videos 📷 used
- Give a Pull Request 🩹 with the message 📩`Developed Script for respective_task` and mention how you have given the argument parameters to run the script in the description
- Once approved, work on documenting every block of code if not every line of your script
- Add a `README.MD` file with appropiate description **[Please ensure you properly cite any research paper or blog you have taken direct reference from]**
- Give another Pull Request 🩹 with a message 📩 : `Documentation Updated`
- Once merged and no other changes is required:
- Move on to working on a new ☀ task
- Look for better methods to improve 🥇 this or any other existing task
- Try to curate a custom dataset 🧰 for your task or anyone of the task
- Propose a new task!

- Contribute to the Documentation of the project in terms of [ReadMe](https://github.com/smaranjitghose/DeepPixel/master/README.md), GitHub Pages📟 inside the `docs` subdirectory or working on sphinx for the documentation of the package

- For the completed issues, Use TensorFlow.JS for building client-side web apps 🌐
- For some of the issues, Use TensorFlow Lite along with Flutter to make mobile apps 📱
- Work on structuring the entire work in the form of a python pacakge 📦
- Fix any bugs you find! 🐛🔨

### How to contribute:

![RGSOC'20](https://img.shields.io/badge/RGSOC-20-red)

**1.** Fork [this](https://github.com/smaranjitghose/DeepPixel) repository.

**2.** Clone the forked repository.

```terminal
git clone https://github.com/<your-github-username>/DeepPixel
```

**3.** Navigate to the project directory.

```terminal
cd DeepPixel
```

**4.** Create a new branch.

```terminal
git checkout -b <your_branch_name>
```

**5.** Make changes in source code.

**6.** Commit your changes.

```terminal
  git add .
  git commit -m "<your_commit_message>"
```

**7.** Push your local branch to the remote repository.

```terminal
git push -u origin <your_branch_name>
```

**8.** Create a Pull Request!

**Congratulations!** You have just made your contribution to [DeepPixel](https://github.com/smaranjitghose/DeepPixel) project.

## Skills Required: 💪

- Python(Mandatory)
- Git(Mandatorty)
- Linux Command Line(Mandatory)
- Elementary Knowledge of Deep Learning or Computer Vision (Mandatory)
- Ability to use TensorFlow 2.0/PyTorch/Keras/fast.ai (any one is suggested)
- OpenCV(Required)
- HTML,CSS,JavaScript(can be picked up on the go)
- Dart and Flutter (can be picked up on the go)
- [TensorFlow.JS](https://www.tensorflow.org/js)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Sphinx](https://www.sphinx-doc.org/en/master/)

**And above all the willingness to learn and contribute!**

## Resources to get started with: 🖊

- [PyImageSearch](https://www.pyimagesearch.com/)-One of the best places on the Internet for anything on Computer Vision
- [Fast.ai](https://www.fast.ai/)-Deep Learning course+Forum for any level that you will fall in love with!
- [MIT:Intro to Deep Learning](http://introtodeeplearning.com/) - shows you more real world applications as you dive into deep learning
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/index.html)
- [MIT Deep Learning](https://deeplearning.mit.edu/) - From self driving cars to Alpha fold..they help you dig into the latest in the field..Lex Fridman's podcasts are just 🔥🔥
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Most popular MOOCs that is used by thousands all over the world to get started with Deep Learning


# Contributors :

[![](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/images/0)](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/links/0)[![](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/images/1)](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/links/1)[![](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/images/2)](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/links/2)[![](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/images/3)](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/links/3)[![](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/images/4)](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/links/4)[![](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/images/5)](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/links/5)[![](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/images/6)](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/links/6)[![](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/images/7)](https://sourcerer.io/fame/smaranjitghose/smaranjitghose/DeepPixel/links/7)


<h3 align="center"><b> 👨 Project Admin</b></h3>


<p align="center"><img width=20% src="https://avatars2.githubusercontent.com/u/46641503?v=4"></p>
<h4 align="center"><b> Smaranjit Ghose</b></h4>


<h3 align="center"><b>  Project Maintianer </b></h3>

- [Suhrid Datta](https://github.com/suhriddatta)

## Community:

    We would love to hear from you! We communicate on the following platforms:

[![Join the chat at https://gitter.im/opendeeppixel/community](https://badges.gitter.im/opendeeppixel/community.svg)](https://gitter.im/opendeeppixel/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## License: 📜

[MIT License](https://github.com/smaranjitghose/DeepPixel/blob/master/LICENSE)
