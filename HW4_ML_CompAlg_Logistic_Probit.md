ML Computational Algorithms: FS and IRLS for Logistic and Probit (only
FS)
================
Courtney Jacks,Chloe Mattila, and Yao Xin
2024-11-07

1.  Fit both Fisher Scoring and IRLS routines and compare to GLM
    function in R for Logistic Regression Model.

First We also make a data frame for the build in glm function of the R
software. We can generate a random data set.

``` r
set.seed(051221)
n<-100
x<-rnorm(n)
X<-cbind(1,x)
beta<-c(-1,1)
eta<-X%*%beta
mu<-exp(eta)/(1+exp(eta))
y<-rbinom(n,1,mu)
```

Next, we will use glm function to perform a Binomial regression model
using the data generated data set and output a summary of the result
called fit.

``` r
fit<-summary(glm(y~x,family="binomial"))
fit
```

    ## 
    ## Call:
    ## glm(formula = y ~ x, family = "binomial")
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  -0.8934     0.2409  -3.708 0.000209 ***
    ## x             0.7852     0.2470   3.179 0.001476 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 126.84  on 99  degrees of freedom
    ## Residual deviance: 114.87  on 98  degrees of freedom
    ## AIC: 118.87
    ## 
    ## Number of Fisher Scoring iterations: 3

Create the initial values for beta hat and eta for IRLS. Also calculate
the number of iterations

``` r
betahat<-c(0,0)
eta<-log((y+.5)/(1-y+.5))  # Empirical logit inits for eta
eps<-100
iter<-0
```

Using a while loop we can now use Iteratively re-weighted least squares
(IRLS) method to update the coefficients.

``` r
# IRLS
while(eps>1e-6){
  iter<-iter+1
  beta_old<-betahat             # Store current betahat
  eta<-X%*%betahat              # Linear predictor
  mu<-c(exp(eta)/(1+exp(eta)))  # Mean function
  W<-diag(mu*(1-mu))          # For bin with canonical log link, W=diag(mu*(1-mu))
  z<-log(mu/(1-mu))+(y-mu)*(1/(mu*(1-mu)))      #Z=Xbeta+Dstar^{-1}%*%(y-mu)
  betahat_irls<-betahat<-solve(t(X)%*%W%*%X)%*%t(X)%*%W%*%z 
  # WLS fit with z as response, eta as mean, and W^-1 as variance
  covb_irls<-solve(t(X)%*%W%*%X)         # Cov(betahat)
  eps<-max(abs(betahat-beta_old))
  print(iter)
}
```

    ## [1] 1
    ## [1] 2
    ## [1] 3
    ## [1] 4
    ## [1] 5

Now lets create the initial values for beta hat and eps for FS. Also
calculate the number of iterations

``` r
betahat<-beta<-c(0,0)
eps<-100        # Stopping value
iter<-1        # Iteration count
```

Using a while loop we can now use FS function below to update the
coefficients.

``` r
while(eps>1e-6){
  eta<-X%*%betahat
  mu<-c(exp(eta)/(1+exp(eta)))       # Update mu
  v<-mu*(1-mu)                       # variance function
  V<-diag(v)
  Iinv<-solve(t(X)%*%V%*%X)          # Inv of Fisher Inv = Asym Cov(betahat)
  u<-t(X)%*%(y-mu)                   # score function, U(beta)
  betahat<-betahat+Iinv%*%u          # Fisher scoring update 
  eps<-max(abs(beta-betahat))        # If current and old beta are "close" then stop
  betahat_fs<-beta<-betahat          # Otherwise update old beta value
  covb_fs<-diag(Iinv)                # Var of betahat
  print(iter)
  iter<-iter+1                       # Update iteration counter
}
```

    ## [1] 1
    ## [1] 2
    ## [1] 3
    ## [1] 4
    ## [1] 5

We can now output the intercept and SEs of the IRLS and FS function and
compare to the fit glm function results.

``` r
glm_ests<-rbind(c(coef(fit)[1:2]),c(sqrt(diag(vcov(fit)))))
IRLS_ests<-rbind(c(betahat_irls),sqrt(diag(covb_irls)))
FS_ests<-rbind(c(betahat_fs),sqrt(covb_fs))

row.names(IRLS_ests)<-row.names(glm_ests)<-c("Estimates", "SEs")
colnames(IRLS_ests)<-c("(Intercept)","x")
colnames(FS_ests)<-c("(Intercept)","x")
rownames(FS_ests)<-c("Estimates","SEs")

# Print results and compared to glm function
glm_ests
```

    ##           (Intercept)         x
    ## Estimates  -0.8933688 0.7852032
    ## SEs         0.2409300 0.2469749

``` r
IRLS_ests
```

    ##           (Intercept)         x
    ## Estimates  -0.8933688 0.7852032
    ## SEs         0.2409427 0.2469981

``` r
FS_ests
```

    ##           (Intercept)         x
    ## Estimates  -0.8933688 0.7852032
    ## SEs         0.2409427 0.2469981

We can see using our self-written IRLS and FS function we got exactly
the same results as the glm function.

2.  Fit both Fisher Scoring Routine only and compare to GLM function
    in R. R code for the FS algorithm for Probit.

``` r
set.seed(021417)
n<-100
x<-rnorm(n)
beta<-c(-1,1)
X<-cbind(1,x)
eta<-X%*%beta
mu<-pnorm(eta)  # Probit link
y<-rbinom(n,1,mu)
fit<-glm(y~x,family=binomial (link="probit"))
```

Now lets create the initial values for beta hat and eps for FS logistic.
Also calculate the number of iterations.

``` r
mu<-rep(.5,n)
eta<-rep(0,n)
betahat<-beta<-c(0,0)
eps<-1e6                      # Stopping value
iter<-1                       # Iteration count
```

We can see using our self-written FS function which uses a while loop to
update the coefficients for the logistic.

``` r
while(eps>1e-6){
  eta<-X%*%betahat
  mu<-c(pnorm(eta))             # Update mu
  v<-mu*(1-mu)                  # variance function
  Vinv<-diag(1/v) 
  D<-X*c(dnorm(eta))            # Dmu_dbeta=X*dmu_deta = diag(dmu_deta)%*%X <--NOTE
  Iinv<-solve(t(D)%*%Vinv%*%D)  # Inv of Fisher Info = Asym Cov(betahat)
  u<-t(D)%*%Vinv%*%(y-mu)       # score function, U(beta)
  betahat<-beta+Iinv%*%u        # Fisher scoring update    
  eps<-max(abs(beta-betahat))   # If current and old beta are "close" then stop
  betahat_fs<-beta<-betahat     # Otherwise update old beta value
  covb_fs<-diag(Iinv)           # Var of betahat
  print(iter)
  iter<-iter+1                  # Update iteration counter
}
```

    ## [1] 1
    ## [1] 2
    ## [1] 3
    ## [1] 4
    ## [1] 5
    ## [1] 6

We can now output the intercept and SEs of the FS function and compare
to the fit glm function results for the logistic regression with link
probit.

``` r
glm_ests<-rbind(c(coef(fit)[1:2]),c(sqrt(diag(vcov(fit)))))
FS_ests<-rbind(c(betahat_fs),sqrt(covb_fs))

colnames(FS_ests)<-c("(Intercept)","x")
rownames(FS_ests)<-c("Estimates","SEs")

# Print results and compared to glm function
glm_ests
```

    ##      (Intercept)         x
    ## [1,]  -0.7863476 0.8041803
    ## [2,]   0.1596658 0.1857887

``` r
FS_ests    
```

    ##           (Intercept)         x
    ## Estimates  -0.7863460 0.8041782
    ## SEs         0.1596688 0.1857938

We can now output the intercept and SEs of the FS algorithm and compare
to the fit glm function results.
