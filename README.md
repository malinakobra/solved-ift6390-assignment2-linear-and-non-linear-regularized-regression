Download Link: https://assignmentchef.com/product/solved-ift6390-assignment2-linear-and-non-linear-regularized-regression
<br>
<h2>1 Linear and non-linear regularized regression</h2>

<h3>1.1        Linear Regression</h3>

Let’s consider a regression problem for which we have a training dataset <em>D<sub>n </sub></em>with <em>n </em>samples (input, target):

<em>D</em><em>n </em>= <em>{</em>(<strong>x</strong>(1)<em>,t</em>(1))<em>, …, </em>(<strong>x</strong>(<em>n</em>)<em>,t</em>(<em>n</em>)<em>}</em>

with <strong>x</strong><sup>(<em>i</em>) </sup><em>∈ </em>R<em><sup>d</sup></em>, and <em>t</em><sup>(<em>i</em>) </sup><em>∈ </em>R.

The linear regression assumes a parametrized form for the function <em>f </em>which predicts the value of the target from a new data point <strong>x</strong>. (More precisely, it seeks to predict the expectation of the target variable conditioned on the input variable <em>f</em>(<strong>x</strong>) <em>‘ </em>E[<em>t|</em><strong>x</strong>].)

The parametrization is a linear transformation of the input, or more precisely an <em>affine </em>transformation.

<em>f</em>(<strong>x</strong>) = <strong>w</strong><em><sup>T</sup></em><strong>x </strong>+ <em>b</em>

<ol>

 <li>Precise this model’s set of parameters <em>θ</em>, as well as the nature and dimensionality of each of them.</li>

 <li>The loss function typically used for linear regression is the quadratic loss:</li>

</ol>

<em>L</em>((<strong>x</strong><em>,t</em>)<em>, f</em>) = (<em>f</em>(<strong>x</strong>) <em>− t</em>)<sup>2</sup>

We are now defining the <strong>empirical risk </strong><em>R</em><sup>ˆ </sup>on the set <em>D<sub>n </sub></em>as the <strong>sum </strong>of the losses on this set (instead of the average of the losses as it is sometimes defined). Give the precise mathematical formula of this risk.

<ol start="3">

 <li>Following the principle of Empirical Risk Minimization (ERM), we are going to seek the parameters which yield the smallest quadratic loss. Write a mathematical formulation of this minimization problem.</li>

 <li>A general algorithm for solving this optimization problem is gradient descent. Give a formula for the gradient of the empirical risk with respect to each parameter.</li>

 <li>Define the error of the model on a single point (<strong>x</strong><em>,t</em>) by <em>f</em>(<strong>x</strong>) <em>− t</em>. Explain in English the relationship between the empirical risk gradient and the errors on the training set.</li>

</ol>

<h3>1.2        Ridge Regression</h3>

Instead of <em>R</em>ˆ, we will now consider a <strong>regularized empirical risk: </strong><em>R</em>˜ = <em>R</em>ˆ+<em>λL</em>(<em>θ</em>). Here <em>L </em>takes the parameters <em>θ </em>and returns a scalar penalty. This penalty is smaller for parameters for which we have an a priori preference. The scalar <em>λ ≥ </em>0 is an <strong>hyperparameter </strong>that controls how much we favor minimizing the empirical risk versus this penalty. Note that we find the unregularized empirical risk when <em>λ </em>= 0.

We will consider a regularization called <em>Ridge</em>, or <em>weight decay </em>that penalizes the squared norm (<em>`</em><sup>2 </sup>norm) of the weights (but not the bias): <em>L</em>(<em>θ</em>) = <em>k</em><strong>w</strong><em>k</em><sup>2 </sup>= <sup>P<em>d</em></sup><em><sub>k</sub></em><sub>=1 </sub><strong>w</strong><em><sub>k</sub></em><sup>2</sup>. We want to minimize <em>R</em><sup>˜ </sup>rather than <em>R</em><sup>ˆ</sup>.

<ol>

 <li>Express the gradient of <em>R</em><sup>˜</sup>. How does it differ from the unregularized empirical risk gradient?</li>

 <li>Write down a detailed pseudocode for the training algorithms that finds the optimal parameters minimizing <em>R</em><sup>˜ </sup>by gradient descent. To keep it simple, use a constant step-size <em>η</em>.</li>

 <li>There happens to be an analytical solution to the minimization problem coming from linear regression (regularized or not). <u>Assuming no bias (meaning <em>b </em>= 0)</u>, find a matrix formulation for the empirical risk</li>

</ol>

<table width="442">

 <tbody>

  <tr>

   <td width="311"> (1) <strong>x</strong><sub>1</sub>and its gradient, with the matrix <strong>X </strong>= <sup></sup><sub> </sub><sup>.</sup>.<sub>.</sub>(<em>n</em>)<strong>x</strong><sub>1</sub></td>

   <td width="33"><em>… </em>…<em>···</em></td>

   <td width="98">(1)<strong>x</strong><em>d </em>…  and the(<em>n</em>)<strong>x</strong><em><sub>d</sub></em></td>

  </tr>

 </tbody>

</table>

 <em>t</em>(1) 

vector <strong>t </strong>=  … .

     <em>t</em>(<em>n</em>)

<ol start="4">

 <li>Derive a matrix formulation of the analytical solution to the ridge regression minimization problem by expressing that the gradient is null at the optimum. What happens when <em>N &lt; d </em>and <em>λ </em>= 0 ?</li>

</ol>

<h3>1.3            Regression with a fixed non-linear pre-processing</h3>

We can make a non-linear regression algorithm by first passing the data through a fixed non-linear filter: a function <em>φ</em>(<strong>x</strong>) that maps <strong>x </strong>non-linearly to a higher dimensional <strong>x</strong>˜.

For instance, if <em>x ∈ </em>R is one dimensional, we can use the polynomial transformation:        <em>x</em>

 <em>x</em>2 

<em>x</em>˜ = <em>φ</em>poly<em>k</em>(<em>x</em>) =  .<sup>.</sup>. <sub></sub>



         <em>x</em><em>k</em>

We can then train a regression, not on the (<em>x</em><sup>(<em>i</em>)</sup><em>,t</em><sup>(<em>i</em>)</sup>) from the initial training set <em>D<sub>n</sub></em>, but on the transformed data (<em>φ</em>(<em>x</em><sup>(<em>i</em>)</sup>)<em>,t</em><sup>(<em>i</em>)</sup>). This training finds the parameters of an affine transformation <em>f</em>

To predict the target for a new training point <em>x</em>, you won’t use <em>f</em>(<em>x</em>) but <em>f</em><sub>e</sub>(<em>x</em>) = <em>f</em>(<em>φ</em>(<em>x</em>)).

<ol>

 <li>Write the detailed expression of <em>f</em><sup>˜</sup>(<em>x</em>) when <em>x </em>is one-dimensional (univariate) and we use <em>φ </em>= <em>φ</em><sub>poly</sub><em>k</em>.</li>

 <li>Give a detailed explanation of the parameters and their dimensions.</li>

 <li>In dimension <em>d ≥ </em>2, a polynomial transformation should include not only the individual variable exponents <em>x<sup>j</sup><sub>i</sub></em>, for powers <em>j ≤ k</em>, and variables <em>i ≤ d</em>, but also all the interaction terms of order <em>k </em>and less between several variables (e.g. terms like <em>x<sup>j</sup><sub>i</sub></em><sup>1</sup><em>x<sup>j</sup><sub>l</sub></em><sup>2</sup>, for <em>j</em><sub>1 </sub>+ <em>j</em><sub>2 </sub><em>≤ k </em>and variables <em>i,l ≤ d</em>). For <em>d </em>= 2, write down as a function of each of the 2 components of <em>x </em>the transformations <em>φ</em><sub>poly</sub>1(<em>x</em>), <em>φ</em><sub>poly</sub>2(<em>x</em>), and <em>φ</em>poly3(<em>x</em>).</li>

 <li>What is the dimensionality of <em>φ</em><sub>poly</sub><em>k</em>(<em>x</em>), as a function of <em>d </em>and <em>k</em>?</li>

</ol>

<h2>2           Practical Part</h2>

You should include all the python files you used to get your results. It should have a main file (which can be a notebook) that produces the required plots, one after another. Your results should be reproducible! Briefly explain how to run your code in the report.

<ol>

 <li>Implement in python the ridge regression with gradient descent. We will call this algorithm regression_gradient. Note that we now have parameters <strong>w </strong>and <em>b </em>we want to learn on the training set, as well an <em>hyper</em>-parameter to control the capacity of our model: <em>λ</em>. There are also hyper-parameters for the optimization: the step-size <em>η</em>, and potentially the number of steps.</li>

 <li>Consider the function <em>h</em>(<em>x</em>) = sin(<em>x</em>) + 0<em>.</em>3<em>x − </em> Draw a dataset <em>D<sub>n </sub></em>of pairs (<em>x, h</em>(<em>x</em>)) with <em>n </em>= 15 points where <em>x </em>is drawn uniformly at random in the interval [<em>−</em>5<em>,</em>5]. Make sure to use the <strong>same </strong>set <em>D<sub>n </sub></em>for <strong>all </strong>the plots below.</li>

 <li>With <em>λ </em>= 0, train your model on <em>D<sub>n </sub></em>with the algorithm regression_gradient). Then plot on the interval [<em>−</em>10<em>,</em>10]: the points from the training set <em>D<sub>n</sub></em>, the curve <em>h</em>(<em>x</em>), and the curve of the function learned by your model using gradient descent. Make a clean legend. <strong>Remark: </strong>The solution you found with gradient descent should converge to the straight line that is closer from the <em>n </em>points (and also to the analytical solution). Be ready to adjust your step-size (small enough) and number of iterations (large enough) to reach this result.</li>

 <li>on the same graph, add the predictions you get for intermediate value of <em>λ</em>, and for a large value of <em>λ</em>. Your plot should include the value of <em>λ </em>in the legend. It should illustrate qualitatively what happens when <em>λ </em></li>

 <li>Draw another dataset <em>D</em><sub>test </sub>of 100 points by following the same procedure as <em>D<sub>n</sub></em>. Train your linear model on <em>D<sub>n </sub></em>for <em>λ </em>taking values in [0<em>.</em>0001<em>,</em>0<em>.</em>001<em>,</em>0<em>.</em>01<em>,</em>0<em>.</em>1<em>,</em>1<em>,</em>10<em>,</em>100]. For each value of <em>λ</em>, measure the average quadratic loss on <em>D</em><sub>test</sub>. Report these values on a graph with <em>λ </em>on the x-axis and the loss value on the y-axis.</li>

 <li>Use the technique studied in problem 1.3 above to learn a non-linear function of <em>x</em>. Specifically, use Ridge regression with the fixed preprocessing <em>φ</em><sub>poly</sub><em>l </em>described above to get a polynomial regression of order <em>l</em>. Apply this technique with <em>λ </em>= 0<em>.</em>01 and different values of <em>l</em>. Plot a graph similar to question 2.2 with all the prediction functions you got. Don’t plot too many functions to keep it readable and precise the value of <em>l </em>in the legend.</li>

 <li>Comment on what happens when <em>l </em> What happens to the empirical risk (loss on <em>D<sub>n</sub></em>), and to the true risk (loss on <em>D</em><sub>test</sub>)?</li>

</ol>