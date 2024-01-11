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
Data points are also called "Feature vectors"
(bold font) x = (x1, x2, ..., xd) d = dimensionality of the vector
Example: Feature vector for a medical patient: x = (21.4 (age), 6.1 (height), 200 (weight))

Comments:
1. Feature vectors are often in higher-dimensional spaces than we can visualize, e.g., for this dataset the penguins are represented in a 4-dimensional feature space
2. Features might not capture all important aspects of a problem, e.g., here we are only measuring certain properties of penguins

If we need to ever get a subset of a feature vector with dimension d, it can be denoted as:
xi = (xi1, xi2, ..., xid) where i = 1, 2, ..., n

Data Matrix:
(cap and bold) X = (x11, x12, x13, ..., xnd) for n rows and d columns.

Target would be a column from the data matrix:
y = (y1
     y2,   
     y3)
```
# Linear Regression
```
Our first machine learning method!!

Regression:
- Feature vector x (Real-valued)
- Target variable y (real-valued)
- Traning Data: set of pairs (xi,yi) where i=1,2,..,n

Regression in 1-dimension - Set of (xi,yi) pairs
How can we make predicitions for inputs (xnew) that we didn't see in our data?

In linear regression, we want to find the "line of best fit"
Equation of a line: f(x|0) = 0zero (bias) + 01x
Parameters of the model: 0 = (0zero,0one)^T

Example of Linear Regression in 2 Dimensions
Linear Model = equation of a plane = f(x ; q) = q0 + q1 x1 + q2 x2
The parameters of the model: 𝜃 = 𝜃! , 𝜃" , 𝜃2
Linear regression model:
𝑓 𝒙 𝜃 = 𝜃! + 𝜃" 𝑥" + 𝜃$ 𝑥$ + ⋯ + 𝜃% 𝑥%

Key Idea: Find the linear regression model that minimizes the prediction error on our data
Loss Function: Mean Squared Error (MSE)
𝐿(𝜃) = 1/𝑛 * summation from t=1 to n of (𝑦! − 𝑓 𝒙! 𝜃))^2
```
# Learning Linear Models with Gradient Descent
```
How can we find the value of 𝜃 = (𝜃! , 𝜃" , ... , 𝜃d)^T that minimize the loss 𝐿(𝜃)?
Gradient Descent Intuition

In machine learning will want to do this type of "downhill move" in many dimensions, not just one
In general, we dont know much about the "shape" of the loss function.
And gradient descent is a heuristic local search algorithm that uses
the gradient – widely used in machine learning

What is a Gradient?
The gradient of a function of multiple variables is a vector of partial derivatives, one for eahc variable.

The graident vector of L(0) points in the steepest uphill direction of the L(0) surface at point 0
```
