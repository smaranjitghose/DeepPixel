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

- Work on structuring the entire work in the form of a python pacakge 📦
- Fix any bugs you find! 🐛🔨

### How to contribute:

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
