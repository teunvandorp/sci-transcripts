# Selecting models

## Course logistics

### Cancelling portfolio item 2

So far, there hasn't been much feedback opportunities between you,
and I'm kind of worried that this will be getting quite difficult to keep track of,
also based on my experience with the questions.
I will scrap that part, which means that you get one less thing to do.
That doesn't mean that I think it's not important.
Feedback is, in this course, important.
We had on Monday a session where quite a few people
showed up.
They showed something, and then there was some feedback from me.
So I think this learning from feedback is actually a quite important thing
that I wanted to have in this course, because that's also how you will,
yeah, so this thing to, this ability to get feedback from us
is kind of the main advantage that I still think there is
in in-person university education.
You can watch nice content online.
There are brilliant free textbooks, YouTube videos, everything.
So I think the biggest advantage of being here is that we can actually watch what you do,
and then we can give feedback.
So I think that has to be, as we are transitioning to, you know,
knowledge being more and more available, I think that providing feedback is maybe the
most important thing in education in general.
So it's, of course, it's difficult to do, but it's really important that you can also
take the feedback in the right way, that it's not, you know, insulting, or that you don't
feel offended, or that you don't think you're doing something wrong just because we're giving
suggestions. I will try to find a way to somehow evaluate this in future editions of the course.


### Explanation item 11 (publication-quality figure)

I think the others are hopefully clear by now, at least these five.
There was a question on portfolio item number 11 on Monday. The item asks for a publication quality figure,
and so what I mean by that is to have an interpretation, some final result of your analysis that you
show in the form of one figure, and it doesn't mean that you just make a plot in my plot
lib, and I know that you can do that, but I want you to work hard.
I want you to work hard on communicating your findings as clearly as possible, and
I think that's something where I think there's still a bit of room for improvement in general,
so it's not about taking something that you've done out of the context of your Python notebook
and just making one figure that has clear labels, is readable, has a font that is actually
high resolution, has a legend that says what you've done, and has kind of an easy interpretation
for somebody who just wants to look at what you've done and just get your conclusion.
That's what I mean by publication quality, right, so very likely, the first thing that
you will show, there might be some feedback on it that you would maybe have to make some
change, because what I see from people's master theses and internship reports is that this
is really difficult.
A lot of people have never actually been given any feedback on any figure that they ever
made, and making figures is actually one of the hardest things still in scientific publishing,
as I also know.

That's also a reason why we wanted to have this in here, because I'm focusing anyway
so much on interpretation.
We also felt that it's maybe good to start working on figure skills early and not in
the master thesis, right, for the first time.
So that's also the reason that this is here.
The idea isn't to make a quick plot in matplotlib and give me a gif and you're done,
but it's really make a figure as it would be in your master thesis or internship report
about one of the things that you did.
And you can choose which.
This is really a separate thing on its own, and here you will be getting feedback on the
figure itself.
So how does it look like?
Would this be good enough to get published in a journal, for example?
That's why it says publication quality.
It should be something that you could send to a scientific journal, and they wouldn't
send it back to you and say, look, you know, make it in a professional way.

### Items 6,7, and 8

At this point we can do items 6 and 7 because we have
talked at length about regression analysis from all of the submissions so far.
I think the interpretation of the model is the thing that gets the least attention, 
but it is said here, please interpret the results.
There are many, many examples of how to do that in the lecture notes.
We can also do the statistical inference on model parameters using the techniques that
we learned about last week, like the Wald test and bootstrapping and the sandwich.
These are techniques that you can use to do inference on a model parameter, and there are
many examples about that in last week's lecture notes.
Today we're going to do the part where we choose between models.
We are going to actually look at two different ways of doing that.
One is likelihood-based, which is what we're going to do today.
So after today, you're also able to complete item 7.
And this other Bayesian part, we're going to do that next week.
So you can choose which one you want to use there, but you could basically do the item
eight with the material of today.

### Difference to the other items

So as of today, we're able to do these first eight things, and starting from nine, it starts
to, you know, go into the topic of causal inference.
So this would be for the second quarter.
So only this first eight you can do in this first quarter with this material, even though
it's not going to be for the second quarter.
So this would be for the second quarter.
So if you really already know how to do these things, and you really want to start and show
it to me, go ahead.
But just from the content that I showed, after today, we have everything that you need to
do, all of these items, right?
So just to be clear again, this is about choosing between two different models.
We're going to talk about it in a second.
Seven is about doing inference on a parameter, which is what we did last week.
If you remember these tests, the wall test and the sandwich, when we constructed these
confidence intervals.
And six is simply about fitting a model, which is what we did in the first four weeks, pretty
much.
So there are many different types of models.
And I just put some extra stuff in there to make it a bit more sophisticated, so that
you just, that you need to do something that goes a bit beyond just fitting one regression
line.
So having an interaction term there, or some data transformation, or some, you know, generalization.
So I think by now we have seen a few items, six, but still not more than, I think, 15 or 16
the last time I counted.
Yeah, and so these with a star, you can team up in groups if you want to, but I still want
to see an individual regression, at least from every single one of you.
And the idea is also that you cannot do all of these in a team, but you can only do two
of them.
I think that's it.
Two of them in a team, so that you have a kind of a mixture between your individual
work, but maybe also teamwork, if you prefer to team up.
That's also fine.

## Introduction to model comparison

Today we're going to talk about how to decide whether models are good, and 
how to compare models to each other. 

I've seen many of you do things like feature selection, removal of insignificant
regression coefficients, and the like.
So I think many of you have some intuitive idea, or maybe some prior experience about
some of these topics.

But I want to go through them systematically today, and specifically in the context of inference.

Here again, many of the things that you already know seem to come from a context of prediction.
And I want to talk today about, is there a difference between what is a good predictive
model and what is a good inference model? These might not necessarily be the same.

First of all, we'll ask: What is even a
good model? What do we mean by that?
Here, it makes sense to distinguish between prediction and inference. We're going to 
argue that:

 * For prediction: Accuracy matters; correctness may not matter
 * For inference: Correctness matters; accuracy may not matter that much

So for prediction, you could say, well, it's an easy question, right?
The best model is simply the one that predicts the most accurately.
So we have some metrics, some accuracy metric.
It could be something like precision or recall, accuracy ...
there are many different ways to measure model performance.

But we don't have a notion of model _correctness_. We can say that a given model
is very accurate, but we can't say that it's "very correct", for deep reasons that 
we talked about in the first week (remember, inductive inference?) 

In many industrial applications, for example, we may just want to have something that predicts well.
For example, if you just want to predict the weather of tomorrow, then, yeah, then the
most accurate weather app is probably going to be the best one.

For inference, this is different. Here, we start with a scientific question,
like in our cell division example from last week: 
We wanted to know how long cells live, and that's a question.
And if I use a model to answer that question, because I can't measure the thing I'm measuring
directly, for example, then correctness is important, right?
So here, we have the idea from the first week that we have a theory and that it has to be
correct in some sense.
But accuracy, on the other hand, maybe my theory isn't super accurate, right?
That may not be that important.
It is important, actually.
There is some role of accuracy here, which we're also going to see later.
But it's not my primary objective to make a model that predicts something with 90% accuracy.
So, for example, if you work in population health or in econometrics or in psychology,
you won't have models that can predict things with 99% accuracy, right?
So there's no model that I know of that predicts any interesting population health outcome
with an accuracy of 90-point-something percent.

At best, we can have things like elevated, most of the things that you know from the
popular science literature, like, hey, if I eat Mediterranean food, I'm going to live
longer, or something like that.
Or if I...
You know, my grip strength, that's a recent result, right?
So grip strength is apparently very predictive of how well you're aging.
But very predictive in that context means that it explains maybe, I don't even know,
but probably something like 20% of the variation in lifetime, or 10 or less, right?
Or GWAS, genetic genome-wide association studies, where you try to predict diseases from your
genome.
A good GWAS could perhaps explain 2% of the variance in a given trait. There aren't currently
any algorithms that, for instance, would look at your genome and go: "You will get diabetes at 
age 62" and would be 99% accurate on that. Why do you think doesn't that exist (yet)?

> Because it's a really complex... It's a complex system with a lot of interaction.

Yes, in complex systems, like our body, or like our mind, or like our society, these things
are just, like, making any predictions is really difficult.
Even the weather is actually pretty difficult to predict.
We're quite good now.
But predicting the weather next week, for example, is still quite hard, because it's
just a complex system.

So, complex systems limit our ability to make very accurate predictions.
But it doesn't mean that these predictions are necessarily useless.
So the knowledge that, for example, you have a high BMI, and that this
may increase your risk for several diseases substantially, is important and 
useful, even if it doesn't necessarily _predict_ well _when_ or _which_ of these 
diseases will affect you personally. 
We don't need to be very good at predicting something to extract some useful 
information about the world, on models that are really not very accurate at all.

Another example would be from business.
Let's say somebody goes on your website and you want them to buy something from your website.
But you can track users now, of course.
If you go to a website, you have cookies.
You can track the mouse movements.
Right.
You have a lot of information.
And you can use that information to some extent to predict.
who's going to buy something but your prediction is still going to be very bad in most cases right
so you're not going to have any algorithm that looks at the mouse movements and then says oh
i'm 99.9 certain that you're going to buy something that's that's just not going to be
the case my model is if the mouse is over to buy something yeah but if there's even then you might
not click on it right so even then i would argue that you're not going to be a super accurate you're
not going to get a super accurate prediction but that doesn't mean that it's not useful because if
you can if you can identify a feature that separates people from you know if you can
identify something you can change on your website and make people buy two percent more by doing that
it can translate into a lot of money right so this two percent can be important and that's that's what
a lot of companies do they run these tests on their website to see it doesn't matter if i place
my button
here or there or does it matter if i make it green or red or blue and this all these
micro optimizations are driven by statistical models that are all very inaccurate but it's
kind of each change makes makes a difference and that's how you optimize your revenue and
if your numbers are large right so accuracy isn't that important but correctness and for scientific
questions is really important if you are making a statement like my white blood cell lives for
93 days
and i really want that to be approximately the correct answer right and actually in that field
that i um that i um mentioned last last week with these blood cells there were actually different
models that led to very different estimations right so one one group was claiming the cells
live for two months and the other was claiming they live for two years and there's a tenfold
difference between these two things right and and then the question was which model is actually
correct and i worked on that for years trying to figure out which model is actually correct and i
figured it out so correctness here is important now um and the problem is with correctness is
that it's a harder thing than accuracy in some sense you can measure accuracy and you can say
well i'm i'm happy if i have an accuracy of 99 but now the question is how do you measure correctness
and now it's important for you to think back to week one where we had the philosophy of science
because people were thinking about exactly that question
can we measure if if a model is correct
yes
okay if you're falsifying the model that means that we throw away an incorrect model
but that doesn't well it doesn't measure correctness or does it
not really but if it's going to be false
but it can be falsified it's correct now i disagree
with that so there's a model that can be falsified is a model that can be tested and that is what we
call the scientific model but it i mean it can be false then like let's say we try to falsify the
model and it is false then it is not correct by definition so how could we how could we measure
correctness do you know any any measure of model correctness that tells you how correct the model
is scientifically correct
you

already right um so but how do i know that the true model is the true model to begin with
yeah so do you have an answer
so yeah so the point is there is no measure of correctness because we cannot actually ever prove
that the model is correct right if you remember proper and deductive inference and so on so the
point of inductive inference which is what we're doing here was that you can actually never prove
that your inferences are correct
because you're always making assumptions when you're doing your inferences uh like for instance
that the world doesn't change from this data point to the next and these assumptions are not
falsifiable so we even said that inductive inferences isn't even rational right so the
entire basis of that of that model construction isn't even a rational activity strictly speaking
so there is no way to um to measure correctness yes yeah i think of just 

> why isn't correctness just like similar to like a high accuracy

we will see that we will now look at some examples because that's exist so for some
data scientists even they wonder what's even the difference between accuracy and correctness

### A correct and accurate model

Let's consider the following example model:

$$ Y = \alpha X^2 + \epsilon \; \;  ; \; \; \epsilon \sim \mathcal{N}(0,\sigma) $$

We're now going to _simulate_ from this model, which is a way (and perhaps pretty much the _only_ way) to set up a scenario where we have the _correct_ model: we know the model because we define it ourselves, and then simulate data from it.

```{r acccorr}
set.seed(123)
x <- seq( 1, 50 ); y <- 0.2 * x^2 + rnorm( 50, sd=10 )
plot( x, y )
```

Here I set $\alpha=0.2$, but it could have been any other value. Now 
let's fit this function to our data:

$$
E[Y] = \alpha X^2
$$

We know that this model is correct because:

$$ E[Y] = E[\alpha X^2 + \epsilon] = E[\alpha X^2] + E[\epsilon] = \alpha X^2 + 0 $$

So we fit the model to the data, using least squares as a loss function:

```{r acccorrfit}
mdl <- function( x, th ) th * x^2
loss <- function(th) mean((y-mdl(x,th))^2)
fit <- nlm( loss, 1 )
plot( x, y ); lines( x, mdl(x,fit$estimate), col=2, lwd=2 )
```


This is, in some sense, our "dream scenario" in inference: the process that generates the data is exactly captured by the model that
we're fitting.  So we know that the model is correct. We can also see that the model is in fact very accurate: all predictions are very close to the data.


### A correct but not accurate model

How would an example look like, then, of a model that is correct
but not accurate? Could we perhaps change something in our simulation to make
the model less accurate, but still correct?

> increase the noise?

Indeed, let's increase the standard deviation of our noise term in the model from 10 to 100 and see what happens:


```{r acccorrfit2}
x <- seq( 1, 50 ); y <- 0.2 * x^2 + rnorm( 50, sd=100 )
mdl <- function( x, th ) th * x^2
loss <- function(th) mean((y-mdl(x,th))^2)
fit <- nlm( loss, 1 )
plot( x, y ); lines( x, mdl(x,fit$estimate), col=2, lwd=2 )
```


I didn't change the model --- it's still the correct model --- but now the accuracy
is indeed much lower. So this is an example of a model that is correct but not very accurate

> But how does an incorrect but accurate model look like ?

That's exactly our next question!

### An incorrect but accurate model

What we know now is that we can have models that are correct but
not very accurate. The other question is: can we make a model is very accurate but not
correct? 

> I did last semester a research into early Alzheimer's
protection yeah especially if you do that at a young age let's say our age it's extremely rare
yeah so if you just say for everyone you don't have all the data you don't have all the data
assignments it's like over 99% 

Good example! By doing this kind of trivial prediction in a very unbalanced data set, you can indeed get a model that is highly accurate but not correct. 

I'll show another example I prepared for you. We're now going to fit a linear regression model our  simulated data set from earlier, instead of a quadratic model:


```{r acccorr2}
set.seed(123);
x <- seq( 1, 50 ); y <- 0.2 * x^2 + rnorm( 50, sd=10 ) ; 
plot( x, y )
mdl <- function( x, th ) th[2] * x + th[1]
loss <- function(th) mean((y-mdl(x,th))^2)
fit <- nlm( loss, c(1,1) )
lines( x, mdl(x,fit$estimate), col=2, lwd=2 )
```



I know now that the model is incorrect because I know that this model I'm fitting is not
the one used to generate the data. But in fact, the accuracy of the model is actually
not bad. This model is in fact more accurate than the correct model in the previous example, 
because the standard deviation of the error is lower.


### Quantifying model accuracy

So far we measured accuracy in a hand-wavy way, by looking at how far the prediction was from the model curve. Let's look at a more formal metric we can use to measure accuracy for a continuous prediction: the $R^2$ value.


This value is defined as follows:

$$
R^2 = 1 - \frac{\text{MSE}}{\text{MSS}} 
$$


Here MSE stands for "mean squared error" --- you should be familiar with this term by now -- and MSS stands for "mean sum of squares" -- this would be the overall sum of squares of the data _after subtracting the mean_, so

$$
\text{MSS} = \frac{1}{n} \sum_i (x_i - \mu)^2
$$ 

Note that this is _almost_ the estimated variance $\hat{\sigma^2}$ of the data, except that there is a different denominator:


$$
\hat{\sigma^2} = \frac{1}{n-1} \sum_i (x_i - \mu)^2
$$ 

Since the MSE of any reasonable model can not be larger than the MSS, $R^2$ is generally a number between 0 and 1
where a very bad  model that doesn't explain anything would have $R^2=0$ because the variance of
the errors would be exactly the same as the variance of the data. Then as our model improves, our $R^2$ will get larger, as the MSE goes down.

We can get $R^2$ very easily from a least squares fit like the one above, because the MSE is already used as the loss function. So, for example, the MSE of the previous model would be:

```{r}
fit$minimum
```

And the MSS is:

```{r}
mean( ( y - mean(y) )^2 ) 
```

Clearly MSE $\ll$ MSS, so we expect our $R^2$ to be large:

```{r}
1  - fit$minimum / mean( ( y - mean(y) )^2 )
```

Indeed, this is a pretty large $R^2$ for our incorrect model!


If my goal is to predict $y$, perhaps I'll be quite happy with this model. Perhaps it won't generalize very well to, e.g., large $x$ above 50, but as long as I don't move too far away from the training data range, I probably could do worse.

So, there are definitely models that are wrong but accurate!

But now let's ask a more interesting question:
can there ever be a more accurate model than the true model?

> We could fit Lagrange polynomials to the data, this would be very accurate but possibly wrong

I think what you're describing is _overfitting_: by making the model very complex, you can fit the training data arbitrarily well. Methods like Lagrange interpolating polynomials can indeed fit any dataset perfectly, other examples would be e.g. spline functions. 

So yes, we can have a model that
is wrong but also highly accurate
on the training data.

in some sense that is what overfitting means: it's making a model that is more accurate than the correct model. that's one way to define overfitting, although it's often done in different ways (e.g. by using learning curves and comparing the training error of a model to the validation error as a function of the sample size). 
that's exactly what overfitting means even though it's not often defined in that way

so now another question so can we then make a model that is more accurate
than the true model on the test or some independent new data so now I'm asking
about what we call generalization right so here we fit we evaluate the model on the data that was
fitted on
but often in machine learning we wouldn't do that we would have some holdout data right some external
data that we maybe set aside

let's say there is a correct model that I can fit can we make a
model that is incorrect but outperforms the correct model on the holdout data
yeah I guess that if the holdout data happens to have a lot of like what's it called outliers
then it might perform better by some situations yeah but would that be structurally better or by
chance better that won't be by chance by chance better yeah by chance anything can happen right
so I could be lucky I could fit two models and just the one happens to be have a bad day and
fits worse but let's think about it in terms of asymptotics like let's say we make the sample
bigger and bigger so that these effects at some point don't matter anymore so do we is it possible
to make a model that is incorrect but generalizes better than the correct model, if there is a
correct model 

so that's basically what we're gonna ask today and the answer is no right and that is
that is really a fundamental truth that is that is how correctness also matters for prediction
in the end right so in the end there is a relationship between accuracy and
correctness and that is that in statistical theory there is no model
more accurate on held out data, on external data,
than the correct model.

And that's actually the reason for us to even in prediction
seek out correct models in the first place.
The correct model, if there is one, generalizes the best.
There's no model that can generalize better
than the correct model.
It's a very fundamental and important statistical fact,
which people often forget, right?

So for instance, that is one reason why we are trying
to make physical simulations of systems,
like the weather or climate,
because if we have an accurate simulation
that actually has all of the things
that are important in it,
there cannot be any machine learning model
that outperforms that,
because it's just not possible
to outperform a correct model.

So the fact that machine learning models
can sometimes outperform, for instance,
physics-based simulation,
means that maybe the physics-based simulations
aren't good enough.
But if you had a simulation like molecular dynamics,
which is actually very accurate,
and we believe, actually,
almost entirely correct,
then you cannot outperform that by machine learning.
(Not all machine learning practitioners are aware of this fact ...)

So it's actually pretty subtle and interesting, right?
So this link between accuracy and correctness is there,
but only on new data.

## Statistical approaches to model falsification

Okay, so, but now we see that the model's wrong,
so I kind of want to,
I want to quickly go now
into this other part of falsification.
So we can fit this model,
and we can say it fits very well,
and we can also test it on held-out data,
and also say it fits very well,
but it doesn't prove that it's the correct model,
because we can never prove that we have the correct model.
We can prove, however, that we have an incorrect model
by trying to falsify the model.

So how do you, how could we falsify this?
Like, we look, we see that something's wrong.
Like, how do you see that this model's wrong, actually?
That is probably not the correct model.
There's something weird about it, yeah.
You expect the data points to be
about normally distributed around the model,
and it's very clearly showing, like,
not normally distributed.

Yeah, but what exactly,
like, if I would plot the distribution of the residuals,
it would probably look like
a normal distribution, approximately.
But there is something weird about the residuals, indeed.
Like, what's weird about them?

> I think if you plot the residuals along the X,
you will see, still see, like, the ...

Yeah, indeed.
So let's look at that carefully.
We have this plot here.

```{r rrplot}
set.seed(123);
x <- seq( 1, 50 ); y <- 0.2 * x^2 + rnorm( 50, sd=10 ) ; 
mdl <- function( x, th ) th[2] * x + th[1]
loss <- function(th) mean((y-mdl(x,th))^2)
fit <- nlm( loss, c(1,1) )
plot( x, y-mdl(x,fit$estimate), ylab="residual")
abline( h=0 )
```


So I think this is the plot that you were referring to.
So what you say is that there is this kind of curve shape.
So the thing is, the regression model itself,
the way that it's fitted, that it is fitted,
ensures that there is no correlation
between residuals and predictor, right?
So even here, there isn't.
There is no correlation whatsoever
between residuals and predictor.
It looks like there is a relation,
but it's not a correlation, right?
So linearly speaking, there is no,
if you would measure the correlation
between X and Y on this plot,
you would get exactly zero.

And that's because the fit works in that way.
I showed that on some previous lecture transcript already,
that basically the way that the loss function is optimized
ensures that this correlation is zero.

But there, I mean, correlation doesn't tell you everything
about relationships
between two variables,
because there can be also non-linear relationships.
So what we have here is clearly there is a relationship
between the residuals and the predictor.
It's just non-linear.
So if the regression model,
if the model that we're fitting were correct,
the model assumes that the errors are just independently
added to the prediction of the quadratic model.
So there should be no relationship whatsoever
between the predictor and the residuals.
And here, it looks like there is a relationship
between the predictor and the residuals, right?

So knowing that, how could we now falsify the model?
What would be a strategy?
We can do something with the residuals.
How would you go about it?
There's actually a lot of different ways,
and no standard way.
But there is something, so maybe you
can tell me if you have an idea how you would now
go about constructing a procedure that, yeah.

> Take a data set that only has points between 20 and 30 X,
and then use that to falsify it essentially,
because then it's actually, accuracy would drop a lot.

So you mean you refit the model
on a smaller part of the data,
and then it gets less accurate?

> No, like, you test on held-out data,
but the held-out data you restrict to the 20 to 30 range,
so that it's really different.

What you're saying is, okay, it looks like,
the model is systematically under-predicting at this point,
right?
So now if you could get a lot of new data in this range,
and you could show that you're always predicting too high,
then that also, indeed, demonstrates
that the model's wrong.

Yes, that is perfectly valid.
It, however, requires me to get new held-out data now.
So can we also do something without getting new data,
just on this data directly?


>  You could do cross-validation, and then for each hold-out,
it would look at what will be the parameters,
and if the variance of the parameters over the hold-outs
would be very big, then there's probably something wrong.

I guess that's not gonna be the case here.
If you do cross-validation on this,
the cross-validation accuracy will still be very high,
and the variance of the parameters will still be quite low.
No, I mean cross-validation over the parameters.
So if you construct the parameters in each training,
and then if you have ten folds, for example,
then probably the variance over those ten folds,
what sort of parameter will be pretty big?
I don't think so, no.
I think the variance here over the parameter, I didn't do it.
We can test that.
I think it will be pretty okay here, because the fit is actually relatively stable.
If I remove a couple of data points randomly, at least,
it will not change the fit much.

If I removed only the first data points, then the fit would change more than you
would expect if you were to remove the first ten from a linear fit.
So maybe there's something that we can do.
Yeah.
But it's a bit of a convoluted test, but that would be maybe possible to say,
yeah, you're going to remove selectively a part of the data, not at random,
but let's say over the range of x, and then the variance is maybe higher
than you would expect from a true model.

Okay, yeah, but it's true.
I mean, there's a lot of different strategies.
But let's look at something that we can just do directly on this data,
just without any extra steps like cross-validation or using held out data.
Is there anything that we could do with this data,
to show that?

The model says there should be no relationship between $X$ and $Y$, yet we can clearly see that there _is_ one. In other words, we should not be able to predict anything about the residuals from $X$, but in fact we are. For instance, if our $X$ is near the mean of $X$, we can predict that the residuals will likely be negative.

What we could to to formally show this would be to fit a _non-linear_ model to the data. For example, if we fit a quadratic model and this turns out to explain the variance in the residuals quite well, that would be a sign that something's wrong: no model --- no matter how complex --- should be able to explain anything
about the residuals from the predictors. Fitting a squared model shows that there might be an issue:

```{r rrplot2}
set.seed(123);
x <- seq( 1, 50 ); y <- 0.2 * x^2 + rnorm( 50, sd=10 ) ; 
mdl <- function( x, th ) th[2] * x + th[1]
loss <- function(th) mean((y-mdl(x,th))^2)
fit <- nlm( loss, c(1,1) )
res <-  y-mdl(x,fit$estimate)
plot( x, res, ylab="residual" )
abline( h=0 )
lines( x, predict( lm( res ~ poly( x, 2 ) ) ), col=2, lwd=2 )
```

This is, indeed, a common strategy to "test" the quality of regression model fits.

### The Wald-Wolfowitz runs test

Here I will do something different, which is a bit simpler.
I alluded to this on Monday, actually.
So I wanted to show how that works.
So what you can also look at consecutive runs of "overprediction" and "underprediction". We can look at this by checking the _sign_ of the residuals: 

```{r}
sign( res )
```

We see that we have a long initial segment where our predictions are all too large, then an even longer one where they're all too small, and then a final one where they're all too large again.

We can summarize this information using a _run length encoding_ of the residual sign:

```{r}
rle( sign(res) )
```

Is there anything unusual about this?
If each residual was really just a random variable with a mean of 0,
then you expect, well, roughly half of them to be minus 1 and roughly half 1,
but also you expect them to switch quite frequently between minus 1 and 1, right?
So if you have a fair coin and you flip it a bunch of times,
then getting 30 times head in a row is quite unusual.

So we can construct a statistical test
by just looking at this very simple statistic,
which is basically just the number of stretches that we can find, right?
So the number of blocks that are all either 1 or minus 1.

So that number here is very small: we have only 3 stretches.
We have one stretch of 1s in the beginning
and a stretch of minus 1s and another stretch of 1s.

So we could think about this number 3 as a statistic and then we could ask,
we have 50 data points here.
If we generate 50 random numbers,
how likely is it to find 3 stretches, right?
Is that very likely? 

We can do that easily by simulation. Let's generate some random data.
The standard deviation and so on isn't important
because we just look at the sign (negative or positive).
And here we see we take the run length encoding
and then we count how many stretches we find.

```{r}
rle( sign( rnorm(50) ) )
```

Quite a few more stretches, and most of them are quite short. We can construct a "null distribution" of this kind of statistic by creating many samples as follows:

```{r rlehypo}
table( replicate( 1000, length( rle( sign( rnorm(50) ) )$lengths ) ) )
```

So as you can see,
we find typically something like 24 or 25 stretches.
This would basically mean that it flips maybe every other time.
The largest amount of stretches we can find is 37.
The highest that we could find is 50.
This would actually mean that we have a perfect alteration
between plus and minus.

On the other hand, the lowest amount of stretches we see in 
these 1,000 simulations is _13_. That is still much higher than the 
_3_ that we have! Our p-value for this observation would be <1/1000
(and it would be much lower if we ran more simulations).

This test is called,
the _Wald-Wolfowitz runs test_. Well, in fact this 
name refers to a mathematical version
of this test where compute the p-value directly 
based on the expected number of runs,
but I just wanted to show the intuition is
you think there are random residuals
and then you construct your expected distribution
of run lengths, and you just see that
it's almost impossible to get only three runs
if your residues would really be independent
of the predictor, right?
So that's a different way of doing that.



Okay?
So we have an incorrect model,
and for incorrect models,
there's typically a way to falsify them,
but to falsify them can be,
can require a bit of creativity
because the aspect that is incorrect
can be different, right?
So here it was the case that it was
this kind of U-shaped residues,
but a different model could just have
something else wrong with it,
like extreme outliers in the data
that the model doesn't account for.
You could not detect that with this test, right?

And that's also why there's no one test
for model incorrectness.
There's no statistical test that uniformly
detects all incorrect models,
because there can be something different wrong
about different wrong models.
It's not all models that are wrong
are wrong in the same way,
and you need to design a test to pick up
different things that can be wrong.
And that's why there's no,
if you run a regression model in R,
there will be no standard correctness test
or something like that,
because there's many different things
that could be wrong.

Okay, so we channeled our inner Popper, performed
a statistical test aiming to falsify our model, and
concluded that our model is indeed wrong. 

### Model extensions can be unnecessary but not wrong

There's one final point that's important to make here.
What happens if we take a model that is correct, 
and then we add something extra to it that is not necessary?
Does that make it wrong?

So in this example, we have,
our model has just a square,
but here I'm going to fit another model
that says that there's also a sine component,
a sine wave here, right?
So I say, well, so the y variable
is a function of x with an intercept and a sine,
which are both not in a true model:

$$ E[Y] = \alpha X^2 + \beta \sin(X)  $$

And now I get this fit:


```{r acccorr3456}
set.seed(123)
x <- seq( 1, 50 ); y <- 0.2 * x^2 + rnorm( 50, sd=100 ) ; 
plot( x, y )
mdl <- function( x, th ) th[1] + th[2] * x^2 + th[3]*sin(x)
loss <- function(th) mean((y-mdl(x,th))^2)
fit <- nlm( loss, c(1,1,1) )
lines( x, mdl(x,fit$estimate), col=2, lwd=2 )
1  - fit$minimum / mean( ( y - mean(y) )^2 )
```

Let's see how this compares to a fit without the extra term:
```{r}
mdl <- function( x, th ) th[1] + th[2] * x^2
loss <- function(th) mean((y-mdl(x,th))^2)
fit <- nlm( loss, c(1,1) )
1  - fit$minimum / mean( ( y - mean(y) )^2 )
```

As expected, I improved my accuracy (even though by a tiny amount).
But the interesting question is now: is this model wrong?

And so counter-intuitively, in the sense of statistics,
this wouldn't be called a wrong model,
because I just said something like,
my,
my function y
is alpha times x squared, right?
So that is the same as before,
but plus also beta times the sine of x.
So this is the model I'm fitting.
The model I'm generating
has beta equals zero.
So in the model I'm generating,
you could say I'm using this model,
I'm just setting beta to zero.

That's why  this model isn't really incorrect: I didn't 
say anything about what beta is.
Beta can be any number.
Alpha can be any number.
And so this model, strictly speaking, isn't incorrect,
because I allow that beta is zero.

If I want to make this an incorrect model,
I would have to write something like,
I also expect that,
I also expect that beta is nonzero, right?
So if I would add this restriction:
$$ E[Y] = \alpha X^2 + \beta \sin(X) \, \, \, , \beta \neq 0 $$
then this model would be incorrect, yeah?
But if I don't do that,
then the model is actually still correct.


That's a distinction that will become very important
in the second part of the course
when we do causal inference,
because also there we are, you know,
really interested in distinguishing
between correct and incorrect models.
And so we don't call this model incorrect.
We have a different way of describing that,
is that we don't like unnecessary complexity, right?
So if something isn't necessary,
then I rather don't want to have it in the model.

But that's what we call parsimoniousness, right?
So there's many different models
that explain the same data.
Many of them can be correct.
And if that's the case,
then I want the one that is the simplest.
I want the one that has the fewest terms,
the fewest parameters,
and we call it parsimoniousness.

So that's also something that we're interested in in inference,
is not only correctness,
because I could also try to make a correct model
by just throwing all kinds of terms in there
and then just hoping that the ones that are actually true,
they are also in there, right?
Then I also have a correct model
just in a scattershot approach
and just putting all possible mathematical equations
in my model and just fit it.

So that's clearly not what we want to do,
so we need this other concept
as well to avoid that we are overcomplicating things.
And that's called parsimoniousness.

Let's summarize that. In inference, we're concerned with two concepts:

 * _Correctness_ - our model should pass falsification attempts;
 * _Parsimoniousness_ – our model should not be more complicated than necessary.

 We're not really concerned with _accuracy_ per se, although we'll see after the break that there's an intimate relationship between accuracy and correctness that does make this aspect interesting for us.

_(Break)_ 

## Model selection by confidence intervals

So let's continue. What I want to do in the second part
is now talk about model selection
so we saw how models can be good or bad
in different senses
one is accuracy and the other is correctness
you already know that we can't measure correctness directly
we can measure accuracy
so we need to use these tools that we have
even if we know that they don't allow us
to show that the model is correct
what we can do is that we can compare models to each other
and that's the basis of statistical model selection
that you are looking at multiple models
and you're trying to make a decision
which model is correct
which is better in some sense

As I already said last week
one very immediate way to do that
is just by looking at confidence intervals of parameters
which we know already how to do

For example, earlier we had this example here
and there we had a regression model
where we had the fuel efficiency
the miles per gallon of a car
predicted as a function of the transmission type
automatic or manual
and the weight of the car
and we had this done using a four parameter model
that contained an interaction term
so the interaction between transmission type and weight.

So the full model was this, let's call it $M_1$:

$$
E[\text{mpg}] = \theta_1 + \theta_2 \text{am} + \theta_3 \text{wt}
 + \theta_4 \, \text{am} \, \text{wt}
$$

so this interaction term $\theta_4 \, \text{am} \, \text{wt}$ 
was a bit annoying
because it made the model very difficult to interpret – 
 simple model can be seen as a line
that you fit through a data cloud
but now with this interaction term
you have actually two lines
that cross each other at some point
so it makes it a bit difficult to interpret. 


We might be wondering whether this interaction term, 
which makes interpretation much harder,
is this really necessary.

We may prefer this simpler model, which is a more _parsimonious_ one
that can also be easier interpreted --- let's call it $M_0$:

$$
E[\text{mpg}] = \theta_1 + \theta_2 \text{am} + \theta_3 \text{wt}
$$

so that's a typical model selection question here
we have a model that is a bit more complicated
and we maybe would like to have it
and we want to see if we can maybe simplify it
so in this model selection question
I'm making a choice between two models
where one is basically a special case
of the other, right?
So you could view the model
that does not have an interaction term
as this one with $\theta_4=0$.

We refer to such model pairs as _nested models_ -- in our case, $M_0$ is nested in $M_1$. Suppose we want to make a choice between those. That's equivalent to deciding: is $\theta_4=0$ (favouring the more parsimonious $M_0$) or not (favouring the more complex $M_1$)?

### More complex models are more accurate on the training data

So first of all let's look at
predictive capacity of the model.


```{r mtcinteract22}
with( mtcars, {
  model <- function(x1, x2, th){
  	th[1] + th[2]*x1 + th[3]*x2 + th[4]*x1*x2 }
  loss <- function(x1, x2, y, th){
  	mean( ( y - model(x1, x2, th) )^2 )
  }
  loss_mtcars <- function(th){
    loss( am, wt, mpg, th )
  }
  fit <- nlm( loss_mtcars, c(1,1,1,1) )
  1 - loss_mtcars(fit$estimate)/var( mpg )
} )
```

So the predictive capacity here is r square value
which I'm computing as 0.83 so it's pretty okay.
So it's pretty okay.
We have 84% as variance explained.
And so one thing that we could do is
when we remove the interaction term from the model
to see if that actually makes a difference, right?
So we could just fit a different version of that model
with that interaction term removed here.

```{r mtcnointeract}
with( mtcars, {
  model <- function(x1, x2, th){
  	th[1] + th[2]*x1 + th[3]*x2 }
  loss <- function(x1, x2, y, th){
  	mean( ( y - model(x1, x2, th) )^2 )
  }
  loss_mtcars <- function(th){
    loss( am, wt, mpg, th )
  }
  fit <- nlm( loss_mtcars, c(1,1,1) )
  1 - loss_mtcars(fit$estimate)/var( mpg )
} )
```

And we see indeed that when we do that
then the r square value decreases from 0.83 to 0.73.
So in terms of accuracy on the training data
the model with interaction term is quite a bit better
than the model that does not have the interaction term.
But now I said on the training data
so that's the catch here.
So I'm never going to get the worst fit on the training data
by making the model more complicated, right?
So a more complicated model will always have
as long as the fitting is stable and everything.
That doesn't have to be.
It's assumed that the fitting works.
Then a more complicated model is always going to be
as least as good as the simpler version
because it's nested in there, right?
So I can always get the simpler one back
by just setting one of the coefficients to zero.

So it means that it would be strange
if the more complicated model fits worse.
This can actually happen
but then it's often for numerical reasons
because my optimizer doesn't find optimum anymore
because my dimensionality gets too high.
But from a fundamental point of view, you 
can never make a model worse on the training data
by making it more complex.
It's neural networks that can work
but then you're working in very high dimensions
where the optimization becomes a really big issue.
But in this low dimensional models
where we have good optimization algorithms,
that generally doesn't happen.

So this kind of, okay, it makes the model more accurate
but also we expect any change that we make to the model
to make it more accurate.
So then the question is,
when do we ever stop here, right?
So somehow we need to take the complexity
of the model into account in that decision that we're making.


Another way to phrase the same thing is: we can't select
between $M_0$ and $M_1$  just fitting $M_1$ and checking if $\theta_4=0$, because that will almost never be the case – even if $\theta_4=0$ really holds in the population, it's unlikely to be the case in any given sample. 

Instead, we need to use some technique like
standard error or confidence interval
to make that decision.
So we learned how to do that last week.

So now let's do a Wald test.
That's something that we learned last week.
To do that, so a Wald test is one way
that we can use to get a confidence interval
for all the model parameters.
In this case, we have four different parameters. Let's fit that model.

```{r mtcinteractwald}
model <- function(x1, x2, th){
	th[1] + th[2]*x1 + th[3]*x2 + th[4]*x1*x2 }
loss <- function(x1, x2, y, th){
	mean( ( y - model(x1, x2, th) )^2 )
}
loss_mtcars <- function(th){
  loss( mtcars$am, mtcars$wt, mtcars$mpg, th )
}
fit <- nlm( loss_mtcars, c(1,1,1,1), hessian=TRUE )
```

Recall that we can do a Wald test by getting the Hessian matrix
from our optimizer and inverting it, and so on.
Inverting the Hessian matrix.

```{r}
FI <- solve( fit$hessian )
FI
```

And then we can look at the diagonal items
of the Hessian matrix, which are the variances of each.
This is now the Fisher information matrix,
so the inverse of the Hessian.
And the diagonal items of that are the variances
of each parameter on its own.



To obtain a 95% confidence interval for each of the model parameters.
I'm taking the estimated vector,
and I'm subtracting 1.96 times the standard deviation
from it to get my lower bound,
and I'm adding 1.96 times standard deviation
to get the upper bound,
and that gives me my confidence interval.

```{r}
cbind( fit$estimate-1.96*sqrt(diag(FI)), fit$estimate+1.96*sqrt(diag(FI)) )
```

So here we look at the results.
So these are my confidence intervals for the four thetas.
This is the interval for $\theta_1$, $\theta_2$,
$\theta_3$, and $\theta_4$, okay?
And this is what we concluded in $\theta_4$, specifically.
So what would we conclude from this analysis now?
Do we need the theta four?
Yeah?

> Well, the value of zero doesn't fall
within the 95% confidence interval,
so it does influence the fit significantly.

Yeah.
So based on the confidence interval,
if it was a hypothesis test, then we would say,
well, we really exclude based on this
that the value is zero.
At the 95% confidence interval level,
it doesn't fall within this range.
So it doesn't look like we can make our lives easier
by kicking this out, okay?

So that is the result.
That's an example.
So if this confidence interval had included the point zero,
then we could maybe have argued,
well, it may have a non-zero value,
but it could also be a zero value,
so we could just remove that from the model
to make it easier, okay?


So that is one approach to model selection.
By looking at...
Nested models, and that's especially relevant
when you start with a big model.
Let's say you have a data set,
and you have a ton of features,
and you just put all of your features in your model
as a first approximation,
then often you're gonna look at this
to see if there's any features that you can kick out
to make the model simpler,
and then maybe generalize better,
make it more stable, more comfortable, and so on, okay?
So this gives you, basically, model selection,
for free, if you know how to do inference already,
then you can also do model selection,
but only for the case of nested models.

### Non-nested models

So there are things that we cannot do with this approach,
like, for example, we talked about the importance
of data transformations a lot.
You could ask sometimes, is it better to fit a model
on a log-transformed data or on the normal data, right?
So this would not be a question that you could answer
using this approach, because these two models aren't nested.
They're actually fitted on different data, right?
So you can't compare them in this way.
They both have a single parameter, maybe,
but they're just fitted on different representations
of the data, so then you wouldn't be able
to use this approach, right?
So it doesn't get you all the way to model selection.
It just gives you the ability to select
between nested models.

## Leave-one-out cross-validation

The second model selection technique that we're going to discuss is cross-validation,
which you probably have seen somewhere already (for example, in the course "CDS: machine learning" that some of you are currently taking). 
Cross validation refers to the idea that you take your data setting
and split it up into two parts.
And these parts are often called training and validation, or
training and testing, depending on who you ask.
In machine learning we have often three different data sets.
the training set, the validation set, and the testing set.
But in statistics, we often have only two,
and then the validation set gets called testing set sometimes.
But the idea is that you have one set that you train on
and another set that you actually check your model on,
and that's not the same data that you trained on.


So one decision that you make then is how do you split your data, right?
So there's K-fold cross-validation, for example,
where you maybe make 10 different bins,
and then you train on nine bins and you test on the last one.
We're only going to look today at leave-one-out cross-validation (LOO-CV), where
you take each individual data point out of the training set and fit a model on the other points. So, a pseudo-code for LOO-CV would look something like this:

 * We train $n$ models $P_{\theta_{-i}}(y \mid x)$ or $E_{\theta_{-i}}[y \mid x]$, each on all datapoints except the $i$-th.
 * We evaluate the losses $L( y_i - E_{\theta_{-i}}[y \mid x_i])$.
 * We average / sum over / otherwise summarize these losses.

LOO-CV is the most labor-intensive form of cross-validation
because it means that you need to fit basically as many models as you have data points.
So if you have 10,000 data points, it would mean that you fit 10,000 models.
And each time you leave one data point out.
So you train your model on the data set, so this P theta minus I.
So the minus I stands for you drop the I state.
So you leave one data point and you fit the parameter theta on that resulting data.
You do this N times, so for all the samples,
and then you evaluate the loss of the Ith output
based on the prediction of the model that's fitted on everything except that data point.

And then we use this loss.
And not the loss that he optimized the model on.
And so that's already, that says you something about,
this should be higher because the other loss function that we determined is the optimal loss.
It's literally the best loss that you can get from the model by tweaking your parameters.
So this loss is generally going to be higher.

### LOO-CV for linear regression

So let's see how that looks like in code and a linear regression model.
So here we have, I'm here just going to use now the
standard algorithm and we're just using the standard R functions for regression here instead of writing our own, just to make the code shorter:

```{r loocv}
L_i <- rep( 0, nrow(mtcars) ) 
for( i in 1:nrow(mtcars) ){
	fit <- lm( mpg ~ am*wt, mtcars[-i,] )
	L_i[i] <- mtcars[i,"mpg"]-predict( fit, newdata=mtcars[i,] )
}
# R-squared from model fit
summary(lm( mpg ~ am*wt, mtcars ))$r.squared
# R-squared from cross-validation
1 - var(L_i)/var(mtcars$mpg)
```


So I have my losses here, there's now a vector of losses.
I iterate over my data frame.
I fit a linear regression model.
This model fits the prediction of the fuel efficiency
as an interaction of these two variables
and also includes all of the individual components.
So this is a four parameter model.
And by this minus i syntax in r,
which is quite convenient, I can just remove the i-th row
from the data frame.
So I'm going to have here the data frame that
contains n minus 1, so in this case, 31 cars instead of 32.
Now I'm using the predict function in r,
which is available to make a prediction for this other data
point that I removed.
And I'm going to compare that to my actual value.
So these are the things I would now call residuals, actually.
So the differences between the predicted value
and the actual value, but not for the fitted data now,
but for out-of-sample data points.
So I can get the r squared value from the model
fit directly like this, which is on the training data.
And this would be my r squared value from the linear.
And if I want to use the data, I just
take the sum of the squared errors.
Just now the errors are estimated by LOO-CV.

And we can see that indeed the r squared
that are estimated from cross-validation
is quite a bit smaller.
So it goes from 83.3% to 78.5%, which is expected.
So it should perform worse, because I have optimized the model
to produce the highest possible $R^2$ on the training set.
Therefore I should expect the $R^2$ on a different dataset
to be lower.


LOO-CV looks a bit like a jackknife that we talked about last week.
The idea that you remove one data point and then fit
the model is the same idea as in a jackknife.
You get the jackknife estimator of your parameters
by looking at all of these fitted coefficients.
So if you're doing jackknife, then basically you
can get the leave-one-out also for free.
So you could do inference and cross-validation
at the same time by that.

So let's do now also the easier model that does not
have the interaction term.


```{r loocv2}
p_resid <- rep( 0, nrow(mtcars) ) 
for( i in 1:nrow(mtcars) ){
	fit <- lm( mpg ~ am+wt, mtcars[-i,] )
	p_resid[i] <- mtcars[i,"mpg"]-predict( fit, newdata=mtcars[i,] )
}
# R-squared from the model fit
summary(lm( mpg ~ am+wt, mtcars ))$r.squared
# R-squared from cross-validation
1 - var(p_resid)/var(mtcars$mpg)
```

So here we have now the syntax `am+wt`.
So this makes the model where these two predictors
are independent, and they don't have this interaction
between them.
This is now a three parameter model.
But apart from that, the entire other function
is completely the same.
And we can see that we already had a lower R-square value
to begin with, 75% approximately, and it goes down to 69%.


So if we summarize all of these things in one table:

| Model | $R^2$ | LOO-CV $R^2$ |
| --- | --- | --- |
| $M_1$ (`am*wt`) | 0.83 | 0.75 |
| $M_0$ (`am+wt`) | 0.75 | 0.68 |

From this, we see that:

 * Model $M_1$ does better on the training data than model $M_0$
 * Model $M_1$ also does better on held-out data than model $M_0$
 * Both models do better on their training data than on held-out data

Another possible result would be that one model scores higher on the training data and another scores higher on the held-out data. In that case, we would choose the model that does better on the held-out data, as the other may be overfitting.

### LOO-CV for logistic regression

So this cross-validation is something
that can be done with any metric that you
could think of for evaluating any kind of model accuracy.
Doesn't need to be R-squared.
So I want to show a different example here.
Not a linear model, but a generalized linear model:
logistic regression.

We'll use our previous example, where we tried to predict the 
transmission type of a car from its weight:

```{r}
model <- function( x, th ) 1/(1+exp(-th[1]-th[2]*x))
loss <- function( x, y, th ){ mean( 
 - log( model( x, th )^y ) - log( (1-model( x, th ))^(1-y) )
) }
loss_cars <- function( th ) loss( mtcars$wt, mtcars$am, th ) 
fit <- nlm( loss_cars, c(1,1) )
fit
```

We saw earlier that, in this dataset, 
heavier cars tend to have manual transmission,
and lighter cars tend to have automatic transmission.
We fitted this model using the negative log likelihood 
as the loss function.
And the resulting negative estimate for $\theta_1$ confirms that 
heavier cars were more likely to be manual. 


This is the same example from two weeks ago.
So we had this quite strong dependence, this minus 4 log
odds ratio, which is a massive effect,
meaning that heavier cars are much, much less
likely to be automatic.
So that is what this number meant.
So how do we evaluate how good a model like this is?
So we had, for the linear regression model,
we had the R-square measure.
For this kind of model, there is no R-square, because, you know,
the data is no longer continuous.
So we need to come up with something else.
Do you have an idea?
What kind of metric would you use to evaluate the performance
of this regression model?

> Accuracy.

Yeah.
And how do we define accuracy?

> Confusion metrics.

OK, so the confusion metrics that
confuse the true positive, true negatives.
How do we get to this?
That's four numbers.
How do we get one number from those four?

> Precision.

Precision is one number that depends only on two entries
of the confusion metrics.
You set a boundary based on which we predict the final outcome.
Yeah, so that's also an important thing to realize.
So this model.
It actually makes a continuous prediction, right?
So the model doesn't directly predict a class,
but it gives you a log odds ratio, right?
So if the number's positive, it means much more likely
to be automated.
Number's negative means much less likely to be automated.

### The area under the ROC curve (AUC)

So another common metric, which I'm showing here also because we you'll use this in other courses later on,
is this metric, the area under the ROC curve.
So the AUC metric, does anyone, did anyone come across the AUC
before?
Because I just want to recap, there's
quite frequent metric used in this kind of scenario.
So you have a continuous, you have basically a way to rank
your cars, in this case, by the likelihood
that they have an automated transmission.
And now what you want is that the cars that actually
do have an automated transmission
score better on this ranking.
I want to show this ranking itself first.
So let's do that.
So this would be a plot where we just
have the predicted probability of the car being
automatic as a function of whether it's actually
automatic, right?
So we see that for the cars that are actually automatic,
most of them also get a high predicted probability
of being automatic.
And for these ones, most of them get actually
a low predicted probability of being automated.
So a good model here, there are some exceptions.
So this car is actually automatic.
It's automated but gets a very low probability
assigned by this model.
And also this one.

```{r}
with( mtcars, beeswarm::beeswarm( model( wt, fit$estimate ) ~ am ) )
```

So if you want to convert this to a confusion matrix,
what you would have to do is we would
have to decide on a threshold where we say, OK,
above this threshold, I'm going to call it predicted automatic
by the model.
And below, I'm going to predict it manually by the model.
But the question is then which threshold?
And this threshold is going to influence the confusion matrix.
If I pick a very low threshold, I'm going to say, OK, I'm going to predict this model.
If I pick a very low threshold, I will have not so many false negatives,
but maybe many false positives.
If I set the high threshold, I might
have a few false positives, but then also not
many true positives.
So there's a trade-off that I'm making.
So one way to evaluate models here
is to consider all possible choices for the threshold.
So without me making the decision beforehand,
I just basically want to know that these two things are well
separated from each other.
And I really want to see that this
is different than that in the direction that I expect.
So the AUC curve is a method where you basically
take every possible threshold that you could use,
and you compute precision and recall
for each of these thresholds.
So you get an entire curve of how precision and recalls
change when you change the threshold.
And that's the ROC curve.
Which we are now going to look at.

```{r}
with( mtcars, plot(pROC::roc( am ~ model( wt, fit$estimate ) ), print.auc=TRUE) )
```

This curve shows the trade-off between specificity and sensitivity.
If I use a threshold that is very high,
then the specificity is super high,
but also sensitivity is super low.
If I set my threshold very low, sensitivity is going to be bad.
But specificity, high.
I think I said at least one of these things wrong.
But I hope that you get what I mean.

> I think usually we show on the horizontal axis,
the false positive rate, and on the y-axis,
the true positive rate.

That's a different way to plot the same thing; there are quite a few equivalent ways
to define this curve.
Any of these is fine.

Ideally, you want to be on the top left point of this graph,
where you have a specificity of 1 and a sensitivity of 1.
That would be great.
But it's not possible.
So we need to be somewhere on this curve
and make a choice, right?
So our trade-off is by being somewhere on this curve.
But generally speaking, it's better
that this curve is far away from this diagonal line, where
basically, if I would toss a coin for each car,
I would be on this diagonal line.
So this diagonal line is basically
a random, completely useless prediction.
And so the more I'm away from that, the better.
So that's what this AUC curve is.
And so by integrating this number,
integrating under this curve, I get, again,
a value between 0 and 1, where 1 would
be a perfect prediction that makes no errors whatsoever.
And 0.5 would be a random prediction.
And 0 would be a worse-than-random prediction.
How could you make a prediction that's worse-than-random?
It sounds strange.
It's impossible, maybe.
But it is possible, yeah.
By fitting a, like, a tangent number to an opposite?
Yeah.
So you have a good model.
But then you flip the prediction.
And you get something that's worse than guessing, right?
So actually, if you had this, it's often a sign
that you have some mistake in your code somewhere,
where you flip a label, or that you don't assign the labels
to numbers correctly, or something like that.
It happens.
And if it happens, it's often a sign that there's
some bug somewhere.


Like $R^2$ for linear regression models, the AUC might be 
optimistic if we evaluate it on the training data. And again,
we can mitigate that effect by cross-validation.


So for this, we can also do cross-validation by, again,
fitting our model.
So this is a model on all data except for one data point,
making a prediction for the last data point,
collecting these predictions.
And then we draw the AUC curve.


```{r logregfit2preds}
p_pred <- rep( 0, nrow(mtcars) ) 
for( i in 1:nrow(mtcars) ){
	fit <- glm( am~wt, mtcars[-i,], family="binomial" )
	p_pred[i] <- predict( fit, newdata=mtcars[i,] )
}
beeswarm::beeswarm( p_pred ~ mtcars$am )
plot(pROC::roc( mtcars$am ~ p_pred ), print.auc=TRUE)
```

And again, we can see that the AUC is lower, right?
It used to be 0.933, and now it's down to 0.891, because it's evaluated
on held-out data.
So again, this is a better way to assess
the accuracy of the model, because we're not
so prone to overfitting.

The general point here is that you can combine cross-validation
with any possible metric of accuracy.

## Akaike's information criterion (AIC)

In statistics, there's often a choice
between doing something based on simulations, like bootstrapping,
and doing something based on mathematical approximations,
like the Wald test. Cross-validation is a simulation-based method.
There's a mathematics-based counterpart to the cross-validation 
approach that's called the _Akaike's Information Criterion (AIC)_.
Who's heard of that before?

> AIC, it's ringing a bell.

Where did you hear about it?

> In model selection.

The AIC is a method you can use for likelihood-based 
models (i.e., whenever you can also do a Wald test).
The AIC is derived in an interesting way by the following argument.

We first note that the likelihood itself is a measure
of how good the model is. Intuitively, models that assign high likelihoods
(and thus low negative log-likelihoods)
to the data they're fitted to are "better" because they "understand"
the data.
In fact, we've seen before that the negative log likelihood
and the sum of squared errors are proportional
to each other.
So a lower negative log likelihood means better model.


The likelihood itself --- as a number --- has no interpretable meaning like $R^2$ or AUC do. However, there is a meaning behind the _difference between likelihoods_. If $m^*$ is _the true model_, and $m$ is any other model, then the _deviance_ 

$$2\,\text{nLL}_{m} - 2\,\text{nLL}_{m^*}$$

In this sense, the deviance is a measure of "how much worse" the model $m$ is than the model $m^*$.

Deviance means that you deviate from the true model
by making a different model.
So the term deviance would suggest that this number should
generally be positive, because $m^*$ should have the lowest
negative log likelihood.
But actually, we can get negative deviances --- by _overfitting_, which can, as we have seen before, produce models that are more accurate than the true one on training data.

That means that the deviance can be negative.


The AIC is a version of the deviance that is derived to ensure that it (asymptotically) cannot be negative.

The AIC is defined as:


$$
AIC_m = 2\,\text{nLL}_{m} + 2 p
$$

where $p$ is the number of parameters of the model.

The additional term $2p$ has the effect that it "punishes" model complexity. Adding new parameters to an already correct model will therefore lead to a lower $\text{nLL}_{m}$, but not necessary a lower $AIC_m$.

The AIC difference between two models is in fact an estimation of the _out-of-sample deviance_. That is, if 

$$\Delta \text{AIC} = 2\, \text{nLL}_{m} - 2p - (2\, \text{nLL}_{m^*} - 2p^*) \approx
 2\, \hat{\text{nLL}}_{m} -  2\, \hat{\text{nLL}}_{m^*}
$$

where $\hat{\text{nLL}}_{m}$ is the negative log-likelihood one would expect from evaluating the model on a new sample of the same size.

Asymptotically, the $\Delta \text{AIC}$ is always positive: no model can get a better AIC than the true model.
Hence, by correcting the likelihood for the model complexity in this fashion,
I can make a metric where I can basically know
that the best scoring model on that metric
is the correct model.

So this is actually a way to measure correctness
in the sense that if I have the correct model already,
I will know that I have the correct model
by comparing every other possible model to it
because it will not be possible
to out-refer to it.
Because of this penalty, right?
So the way that we normally out-perform the correct model
is by adding additional random stuff
that we don't need to it.
No, that's the...
I mean, for that, we would have to go really deep
into the theory of the AIC and how it's derived,
but the mathematical theorem shows that that cannot be done.
So there's a mathematical guarantee
that as your sample size goes to infinity,
you are unable to out-perform the correct model on AIC.

So this means that we can measure model correctness,
doesn't it?
And actually, that's what you said at the very beginning,
but the problem with this is that we need
to have the correct model.
So if we have the correct model,
then we will not be able to think
that a different model is correct
by using this metric.
But if we just have two models,
and none of these two is correct,
then we know nothing, basically, right?
Because then it could always be that there's yet another...
Yeah, so as long as we don't know for sure
that they have found the correct model,
well, then we have maybe one that has the lowest AIC,
but it still doesn't guarantee that the model is correct,
because it could still be that it's a different one
that we haven't built yet that has an even lower AIC.
So it both measures and doesn't measure
model correctness at the same time.
In a very tricky sense.
That's a bit clear.
So I think this is a nice, amazing theory.
So the theory says that no model
can have a lower AIC than a true model.
The AIC number itself means nothing.
It can be something like 275,000.
It depends on the sample size.
It's completely meaningless.
The only meaningful thing is the difference
between the two.
Between the AICs.
This can be also interpreted:



### Interpretation of AIC differences

For interpretation, see:



| $\Delta$ AIC  to best model| interpretation |
| --- | --- | 
| <2 | roughly equally plausible |
| 3-7 | much less plausible |
| >7 | implausible |

<https://sites.warnercnr.colostate.edu/wp-content/uploads/sites/73/2017/05/Burnham-and-Anderson-2004-SMR.pdf>



If we have an AIC difference of lower than two
between two models,
then we could say they're roughly equally plausible.
So that doesn't really mean that much.
Between three and seven,
we would start to say something like,
oh yeah, it looks like this one model
is actually better than the other.
So maybe less plausible.
And if there's a difference of more than seven,
then most researchers would say,
well, this other model is clearly nonsense.
So we shouldn't use that.

## The connection between cross-validation and complexity penalties

Now let's get to the final part,
which I think is mind-blowing,
is that it looks like we've done
two completely different things.
In the beginning,
we had this held-out data set approach
to establish rankings between models.
In the second part,
we had this complexity penalty
where we just said,
okay, more complex models get punished.
It looks like these two things
are completely different:

 
 * _Training-validation split_: Fit a model to part of the data (training) and evaluate on another part (validation);
 * _Complexity penalty_: Compute adjusted accuracy metrics by "punishing" the inclusion of additional parameters.

Interestingly, these two aren't so different, and a complexity penalty could be seen as a quick-and-dirty approximation to cross-validation.

but they're actually the same.
And that can be seen by considering
that you can use the likelihood itself.
You could also cross-validate on a likelihood, right?
The likelihood itself is a loss function,
the negative log likelihood.
So you can also do cross-validation
on a negative log likelihood
in exactly the same way
as for any other loss function.
So let's do that here.
We have two different models.
We have our likelihood function,
and we do negative log likelihood function.
And we do leave-one-out cross-validation
on both of these models
to estimate their negative log likelihoods
in that fashion.
That works the same way
as for any other metric.
So let's compare them.
So what we see is,
so now we have two different models.
One where we're using the weight as a predictor,
and the other,
where we're using the MPG as a predictor.
We just want to know which one of these two is better, right?
So they're not nested.

Let's compare these two models using LOO-CV nLL as a metric:

```{r logregfit2preds223}
with( mtcars, {
  model <- function( x, th ) 1/(1+exp(-th[1]-th[2]*x))
  loss <- function( x, y, th ){ mean(
   - log( model( x, th )^y ) - log( (1-model( x, th ))^(1-y) )
  ) }

  nll_test_wt <- rep( 0, nrow(mtcars) )
  nll_test_mpg <- rep( 0, nrow(mtcars) )

  for( i in 1:nrow(mtcars) ){
  	loss_cars_wt <- function( th ) loss( wt[-i], am[-i], th )
  	loss_cars_mpg <- function( th ) loss( mpg[-i], am[-i], th )
  	nll_test_wt[i] <- loss( wt[i], am[i], nlm( loss_cars_wt, c(1,1) )$estimate )
  	nll_test_mpg[i] <- loss( mpg[i], am[i], nlm( loss_cars_mpg, c(1,1) )$estimate )
  }

  beeswarm::beeswarm( list(wt=nll_test_wt, mpg=nll_test_mpg), ylab="negative log-likelihood" )

  cat("am ~ wt: nLL =", sum( nll_test_wt ), "\n")
  cat("am ~ mpg: nLL =", sum( nll_test_mpg ), "\n")
} )
```

So the negative log likelihood
of this model based on the weight
is smaller than the one based on the MPG
after cross-validation.
So based on this,
I would pick the model that has the weight
rather than the model that has the MPG
because it assigns a higher likelihood to my data, right?

Let's look again at the entire model fit (without cross-validation) for the "better" mpg model:

```{r mpgaic}
model <- function( x, th ) 1/(1+exp(-th[1]-th[2]*x))
loss <- function( x, y, th ){ sum( 
 - y*log( model( x, th ) ) - (1-y)*log( (1-model( x, th )) )
) }
loss_cars <- function( th ) loss( mtcars$wt, mtcars$am, th ) 
fit <- nlm( loss_cars, c(1,1) )
loss_cars( fit$estimate )

loss_cars_2 <- function( th ) loss( mtcars$mpg, mtcars$am, th ) 
fit_2 <- nlm( loss_cars_2, c(1,1) )
loss_cars_2( fit_2$estimate )
```


Here we see that we get better negative log-likelihoods when evaluating on the 
training data, which again is expected, like for any other loss function.


There's a very interesting connection between all these numbers:



$$
\text{AIC} = 2 \text{nLL} + 2p \underset {n \to \infty}{\longrightarrow} 2 \text{LOO-nLL}
$$


In words: Asymptotically, the AIC is twice
the leave-one-out negative log likelihood.
So as your sample gets larger, these two should converge 
to the same value. 

The reason that this is so interesting is that it suggests we can
_approximate cross-validation without actually doing any cross-validation.

Let's check this here: 

| Model | nLL | LOO-nLL | AIC | 2 LOO-nLL |
| --- | --- | --- | --- | --- |
| am ~ wt | 9.6 | 13.4 | 23.2| 26.74 |
| am ~ mpg | 14.8 | 17 | 33.6 | 34 | 


This doesn't even require the correct model
to be tested: The AIC of _any_ model
converges to twice the leave-one-out negative log likelihood.
This means that you can also use the AIC
as a quick initial approximation for cross-validation that you can
get very quickly while you wait for the results of 
an actual cross-validation.

I think this connection also reveals a deep truth 
about the effect of cross-validation: fundamentally,
cross-validation is something that penalizes complex models,
That's why cross-validation works, and why it is a valid
approach to identify a "correct" model, were we to ever have one.

And now, we also see that there is, in fact, a deep connection between
_accuracy_ (on the validation set) and _correctness_, which wasn't so obvious earlier.

I think it's just amazing, honestly -- it almost makes me emotional.

You can find the proof of this connection between AIC and LOO-CV in this paper, if 
you are interested:

<https://sites.stat.washington.edu/courses/stat527/s13/readings/Stone1977.pdf>


### How the amount of held-out data for cross-validation determines complexity

A last, quick question before we wrap up.
We just now did leave-one-out cross-validation.
What do you think changes when you leave out more than one?
Like, let's say we do 80-20% split,
which is more common.

We now understand,
hopefully, that cross-validation
penalizes model complexity.
What do you expect to change about this penalty when you hold out more data?

> It gets bigger.

Yes, what happens is that if you leave out more data,
you penalize complex models even more.

Something like an 80-20 cross-validation
will be more strict against complexity
than the LOO-CV.

So if you want to be more on the side
of selecting simpler models,
you leave out more data;
if you want to be more on the side
of fitting the best model you can,
you leave out less data.

Asymptotically, all these methods
select the correct model in the end, if the correct model is among those
that are considered.
So with sufficient data, the choice how much of it we 
hold out becomes less relevant.
But we never have infinite data, right? We need to work
with finite, and sometimes small, dataset.

When you decide how much data to leave out,
you're basically deciding
how much you penalize complex models.
Remember that next time
you do a cross-validation. When you 
decide whether to do a 5:1 or an 10:1 split, that choice
can influence how complex the model
is going to be that you choose.

Thanks!

## Test your understanding

1. Which of the following statements about model accuracy and correctness is correct?
   * A model can only be correct if it is very accurate on the training data
   * An incorrect model can be more accurate than a correct model on training data
   * Very complex models on average outperform the correct model on held-out data

1. You fit a linear regression model to data that was generated from a quadratic process. When you plot residuals against the predictor, you observe a clear U-shaped pattern. Describe two different statistical approaches you could use to formally demonstrate that this model is misspecified.

1. Researcher 1 claims: "This data was generated by a process that follows the law $E[Y] = \alpha X + \beta X^2$." Reviewer 2 responds: "You're absolutely wrong! I can prove that the model $E[Y] = \alpha X$ achieves a lower AIC on the latest benchmark data, so your model is incorrect!" Who is right?

1. You are choosing between two nested models: $M_0$ with parameters $\theta_1, \theta_2, \theta_3$ and $M_1$ with parameters $\theta_1, \theta_2, \theta_3, \theta_4$. After computing Wald confidence intervals, you find that the 95% CI for $\theta_4$ includes zero. Which model would you choose, and why?

1. Which of the following statements about cross-validation is incorrect?
   * LOO-CV is more computationally expensive than k-fold CV with k=5
   * Cross-validation can be combined with any loss function or accuracy metric
   * Models generally perform better on held-out data than on training data in cross-validation
   * Cross-validation implicitly penalizes model complexity in a similar fashion as AIC

1. The AIC is defined as $AIC = 2 \cdot \text{nLL} + 2p$, where nLL is the negative log-likelihood and $p$ is the number of parameters. Which advantage does AIC have over nLL when comparing an incorrect model to a correct model?

1. Suppose you compare two models and find that Model A has AIC = 245.3 and Model B has AIC = 241.8. What can you conclude about these models? What if Model A had AIC = 241.0 and Model B had AIC = 241.8?

1. Explain the relationship between AIC and leave-one-out cross-validation (LOO-CV), in at most four sentences.

1. You perform LOO-CV on two models using $R^2$ as your metric. Model 1 achieves training $R^2 = 0.89$ and LOO-CV $R^2 = 0.75$. Model 2 achieves training $R^2 = 0.82$ and LOO-CV $R^2 = 0.81$. Which model would you choose and why?

1. When performing k-fold cross-validation to compare different models, how does the choice of k (the number of folds) affect the complexity of the model that will be chosen?

1. We fit a linear regression model with an intercept, two predictors, and an interaction term. The nLL is 45.2. Calculate the AIC.

1. You fit a regression model and obtain MSE = 12.5. The mean sum of squares of your data is MSS = 50.0. What is the $R^2$ value for this model?

1. You compare three models and obtain the following AIC values: Model A has AIC = 156.3, Model B has AIC = 152.1, and Model C has AIC = 159.8. Interpret these results using the guidelines from the course. Which model would you choose and what can you say about the plausibility of the other models compared to your chosen one?

1. We fit a two-parameter logistic regression model and obtain nLL = 18.6. Using leave-one-out cross-validation, we obtain LOO-nLL = 22.4. Compare this value to its approximation based on the AIC.
