## getting the best values for p, d and q in ARIMA (p,d,q)
autoarima <- auto.arima(x, allowdrift=F) # where x is the data
autoarima
