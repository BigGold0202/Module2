brun_slice = read.csv('brun_slice_mg.csv', na.strings = '')
brun_vi = read.csv('brunch_vi.csv', header = FALSE, colClasses = "character")

for (i in 1:10) {
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

#===========rest==========================================================
brun_slice = read.csv('rest_slice_me.csv', na.strings = '')
brun_vi = read.csv('rest_vi.csv', header = FALSE, colClasses = "character")

for (i in 1:10) {
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