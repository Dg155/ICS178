## Lecture 2 Notes: Linear Regression

# Presenting and Exploring Data
```
The Penguien dataset: 344 penguins, described by 4 real-valued features (attributes)
- bill-length
- bill-depth
- flipper-length
- body-mass
Each penguin is in 1 of 3 classes: Adelie, chinstrap, gentoo

Plotting all the data in a histrogram can already tell us if the data is unimodal (1 peak) or multimodal (multiple peaks)

Scatter Plots - Plotting the values of one feature against another.

* Python Code
import seaborn as sns
sns.set_theme() # apply the default theme
# load the penguins dataset
penguins = sns.load_dataset("penguins")
# generate a scatter plot
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm")

If you wanna get a better idea of your data, you should select all possible combination of 2 features, and make scatter plots for each. 2^num_features

When we talk about classifies, we want to find the boundaries that most accuratley seperate these classes.
These are called "decision boundaries"

We can simplify the 2D plot to only portray 1 feature within a 1D feature space. However 2D will be much better than either single feature for classifying this data.
```
# Mathematical Notation
```
```
# Linear Regression
```
```
# Learning Linear Models with Gradient Descent
```
```
