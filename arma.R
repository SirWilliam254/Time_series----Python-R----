## getting the best values for p and q in ARMA(p,q)
# The best is the one which has the lowest AIC value
 final.aic <- Inf
 final.order <- c(0,0,0)
 for (i in 0:4) for (j in 0:4) {
     current.aic <- AIC(arima(x, order=c(i, 0, j))) # x is the data
     if (current.aic < final.aic) {
         final.aic <- current.aic
         final.order <- c(i, 0, j)
         final.arma <- arima(x, order=final.order)
      }
   }
 final.aic # the best AIC
 final.order # the best p and q values
 final.arma # the model emanating from the best parameters.


