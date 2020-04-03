---
name: Feature request
about: Suggest an idea for this project
title: ''
labels: documentation, script, jupyter, dataset, rgsoc
assignees: ''

---

# Task:
Enhance images taken in low light conditions

# Suggested workflow:

- Inside the ```deeppixel``` directory, create a new sub-directory ```img_undark```__[Please name it appropiately and use [camel_case](https://medium.com/better-programming/string-case-styles-camel-pascal-snake-and-kebab-case-981407998841)]__
- In the first attempt 💭💭 use a [Jupyter notebook] to perform your work.  
- Once you are done give a Pull Request🩹 with the message 📩```Developed Jupyter Notebook for Enhancement of Low Light Images``` , briefing about your approach in the description and add a link of the above notebook in [Google Colab](https://colab.research.google.com/) __[Please ensure you have given access]__ ⛔
- Once merged😎 , build a script for the same task in the ```img_undark ``` directory __(If you are using Deep Learning, ensure that you have saved your trained model and its weights so that in the script you build can simply fetch it instead of training again)
- Use [argparse](https://docs.python.org/3/library/argparse.html) library so that the input image and the output path can be given as arguments in the terminal while running the script
- Update the ```requirements.txt``` file in the root directory of the master branch to ensure any additional modules you have used in present there.
- Make sure you provide sample images/videos 📷 used
- Give a Pull Request 🩹 with the message 📩```Developed Script for Enhancement of Low Light Images``` and mention how you have given the argument parameters to run the script in the description 
- Once approved, work on documenting every block of code if not every line of your script 
- Add a ```README.MD``` file with appropriate description __[Please ensure you properly cite any research paper or blog you have taken direct reference from]__ 
- Give another Pull Request 🩹 with a message 📩 : ```Documentation Updated for Enhancement of Low Light Images``` 
- Now save the model and the model weights, build a single python script that takes in an image and gives us the output (Make sure the model and model weight is properly named for future use) by using your already trained model

# References :
- []() research paper

- The official [implementation]() 

- [An explanatory blog]()



# Additional Tasks: 

-  Curate a custom dataset 🧰 for this task
-  Look for better methods to improve 🥇 this
