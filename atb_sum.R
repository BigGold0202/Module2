# This R code is to perform linear regression of selected attributes

brun_slice = read.csv('brun_slice_mg.csv', na.strings = '')   # read in brunch attribute data
brun_vi = read.csv('brunch_vi.csv', header = FALSE, colClasses = "character") # read in brunch feature importance score 

for (i in 1:15) {                        
  i =                                        
  tmp = na.omit(brun_slice[,c(brun_vi[i, ], 'stars')])      # delete all the na values (a lot of NAs)
  tmp[, 1] = factor(tmp[, 1])
  mod = lm(tmp[ , 2] ~ tmp[ , 1])
  a = summary(mod)$coefficients                     # use a loop to print out the coefficients of corresponding linear regresion
  row.names(a) = levels(tmp[ , 1])
  print(brun_vi[i, ])
  print(a[, c(1,4)])
  cat('\n')
}

i = 1
tmp = na.omit(brun_slice[,c(brun_vi[i, ], 'stars')])
tmp[, 1] = factor(tmp[, 1])
mod = lm(tmp[ , 2] ~ tmp[ , 1])
print(brun_vi[i, ])
print(summary(mod)$coefficients)

#===========rest==========================================================
brun_slice = read.csv('rest_slice_me.csv', na.strings = '')          # read in restaurant attribute data
brun_vi = read.csv('rest_vi.csv', header = FALSE, colClasses = "character")    # read in brunch feature importance score 


for (i in 1:15) {
  tmp = na.omit(brun_slice[,c(brun_vi[i, ], 'stars')])
  tmp[, 1] = factor(tmp[, 1])
  mod = lm(tmp[ , 2] ~ tmp[ , 1])
  a = summary(mod)$coefficients
  row.names(a) = levels(tmp[ , 1])
  print(brun_vi[i, ])
  print(a[, c(1,4)])
  cat('\n')
}

i = 1
tmp = na.omit(brun_slice[,c(brun_vi[i, ], 'stars')])
tmp[, 1] = factor(tmp[, 1])
mod = lm(tmp[ , 2] ~ tmp[ , 1])
print(brun_vi[i, ])
print(summary(mod)$coefficients)