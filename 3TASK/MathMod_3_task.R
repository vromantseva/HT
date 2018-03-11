
library('GGally')
library('MASS')
# install.packages('mlbench')
library('mlbench')
data(PimaIndiansDiabetes)
head(PimaIndiansDiabetes)
#Зададим ядро генератора случайных чисел и объём обучающей выборки.

my.seed <- 234
train.percent <- 0.75
options("ggmatrix.progress.bar" = FALSE)

#Исходные данные: набор PimaIndiansDiabetes
?PimaIndiansDiabetes
head(PimaIndiansDiabetes)

str(PimaIndiansDiabetes)

# графики разброса
ggp <- ggpairs(PimaIndiansDiabetes)
print(ggp, progress = FALSE)
# доли наблюдений в столбце Diabetes
table(PimaIndiansDiabetes$diabetes ) / sum(table(PimaIndiansDiabetes$diabetes ))

#Отбираем наблюдения в обучающую выборку
set.seed(my.seed)
inTrain <- sample(seq_along(PimaIndiansDiabetes$diabetes),
                  nrow(PimaIndiansDiabetes)*train.percent)
df <- PimaIndiansDiabetes[inTrain, ]
# фактические значения на обучающей выборке
Факт <- df$diabetes

#Строим модели, чтобы спрогнозировать Diabetes
#Логистическая регрессия
model.logit <- glm(diabetes ~ pregnant + glucose + pressure + triceps + 
                     insulin + mass + age , data = df, family = 'binomial')
summary(model.logit)


model.logit <- glm(diabetes ~ pregnant + glucose + pressure + triceps + 
                     mass + age , data = df, family = 'binomial')
summary(model.logit)


model.logit <- glm(diabetes ~ pregnant + glucose + pressure +  
                     mass + age , data = df, family = 'binomial')
summary(model.logit)


model.logit <- glm(diabetes ~ pregnant + glucose + pressure +  
                     mass, data = df, family = 'binomial')
summary(model.logit)







# прогноз: вероятности принадлежности классу 'Pos' (диабет)
p.logit <- predict(model.logit, df, type = 'response')
Прогноз <- factor(ifelse(p.logit > 0.5, 2, 1),
                  levels = c(1, 2),
                  labels = c('neg', 'pos'))
# матрица неточностей
conf.m <- table(Факт, Прогноз)
conf.m

# чувствительность
conf.m[2, 2] / sum(conf.m[2, ])

# специфичность
conf.m[1, 1] / sum(conf.m[1, ])

# верность
sum(diag(conf.m)) / sum(conf.m)


#QDA
model.qda <- qda(diabetes ~ pregnant + glucose + pressure + triceps + 
                   insulin + mass + age, data = PimaIndiansDiabetes[inTrain, ])
model.qda

# прогноз: вероятности принадлежности классу 'Pos' (диабет)
p.qda <- predict(model.qda, df, type = 'response')
Прогноз <- factor(ifelse(p.qda$posterior[, 'pos'] > 0.5, 
                         2, 1), 
                  levels = c(1, 2), 
                  labels = c('neg', 'pos')) 




# матрица неточностей
conf.m <- table(Факт, Прогноз)
conf.m

# чувствительность
conf.m[2, 2] / sum(conf.m[2, ])

# специфичность
conf.m[1, 1] / sum(conf.m[1, ])

# верность
sum(diag(conf.m)) / sum(conf.m)









# Подбор границы отсечения вероятностей классов —----------------------------— 

# ROC-кривая =========================================================== 

# считаем 1-SPC и TPR для всех вариантов границы отсечения 
x <- NULL # для (1 - SPC) 
y <- NULL # для TPR 
x1 <- NULL 
y1 <- NULL 
# заготовка под матрицу неточностей 
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2)) 
rownames(tbl) <- c('fact.No', 'fact.Yes') 
colnames(tbl) <- c('predict.No', 'predict.Yes') 
# вектор вероятностей для перебора 
p.vector <- seq(0, 1, length = 501) 
# цикл по вероятностям отсечения 
for (p in p.vector){ 
  # прогноз 
  Прогноз <- factor(ifelse(p.qda$posterior[, 'pos'] > p, 
                           2, 1), 
                    levels = c(1, 2), 
                    labels = c('neg', 'pos')) 
  
  # фрейм со сравнением факта и прогноза 
  df.compare <- data.frame(Факт = Факт, Прогноз = Прогноз) 
  
  # заполняем матрицу неточностей 
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'neg', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'pos', ]) 
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'pos', ]) 
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'neg', ]) 
  
  # считаем характеристики 
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1]) 
  y <- c(y, TPR) 
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2]) 
  x <- c(x, 1 - SPC) 
} 
for (p in p.vector){ 
  # прогноз 
  Прогноз1 <- factor(ifelse(p.logit > p, 
                            2, 1), 
                     levels = c(1, 2), 
                     labels = c('neg', 'pos')) 
  
  # фрейм со сравнением факта и прогноза 
  df.compare <- data.frame(Факт = Факт, Прогноз1 = Прогноз1) 
  
  #заполняем матрицу неточностей 
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'neg', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'pos', ]) 
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'pos', ])
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'neg', ]) 
  
  # считаем характеристики 
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1]) 
  y1 <- c(y1, TPR) 
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2]) 
  x1 <- c(x1, 1 - SPC) 
} 
# строим ROC-кривую 

par(mar = c(5, 5, 1, 1)) 
# кривая 
plot(x, y, type = 'l', col = 'blue', lwd = 2, #qda 
     xlab = '(1 - SPC)', ylab = 'TPR', 
     xlim = c(0, 1), ylim = c(0, 1)) 
lines(x1, y1, type = 'l', col = 'red', lwd = 2, #logit 
      xlab = '(1 - SPC)', ylab = 'TPR', 
      xlim = c(0, 1), ylim = c(0, 1)) 
# прямая случайного классификатора 
abline(a = 0, b = 1, lty = 3, lwd = 2) 
# точка для вероятности 0.5 
points(x[p.vector == 0.5], y[p.vector == 0.5], pch = 16) 
text(x[p.vector == 0.5], y[p.vector == 0.5], 'p = 0.5(qda)', pos = 4) 
# точка для вероятности 0.2 
points(x[p.vector == 0.2], y[p.vector == 0.2], pch = 16) 
text(x[p.vector == 0.2], y[p.vector == 0.2], 'p = 0.2(qda)', pos = 4) 

# точка для вероятности 0.5 
points(x1[p.vector == 0.5], y1[p.vector == 0.5], pch = 16) 
text(x1[p.vector == 0.5], y1[p.vector == 0.5], 'p = 0.5(logit)', pos = 4) 
# точка для вероятности 0.2 
points(x1[p.vector == 0.2], y1[p.vector == 0.2], pch = 16) 
text(x1[p.vector == 0.2], y1[p.vector == 0.2], 'p = 0.2(logit)', pos = 4) 


df1 <- PimaIndiansDiabetes[-inTrain, ] 
# фактические значения на обучающей выборке 
Факт1 <- df1$diabetes 

# Logit ========================================================================== 

model.logit <- glm(diabetes ~ pregnant + glucose  +  
                     mass, data = df1, family = 'binomial') 
summary(model.logit) 

p.logit <- predict(model.logit, df1, type = 'response') 
Прогноз <- factor(ifelse(p.logit > 0.5, 2, 1), 
                  levels = c(1, 2), 
                  labels = c('neg', 'pos')) 


# QDA ========================================================================== 
model.qda <- qda(diabetes ~ pregnant + glucose + pressure + triceps + 
                   insulin + mass + age, data = df1) 
model.qda 
# прогноз: вероятности принадлежности классу 'pos' (диабет) 
p.qda <- predict(model.qda, df1, type = 'response') 
Прогноз <- factor(ifelse(p.qda$posterior[, 'pos'] > 0.5, 
                         2, 1), 
                  levels = c(1, 2), 
                  labels = c('neg', 'pos')) 


# считаем 1-SPC и TPR для всех вариантов границы отсечения 
x <- NULL # для (1 - SPC) 
y <- NULL # для TPR 
x1 <- NULL 
y1 <- NULL 
# заготовка под матрицу неточностей 
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2)) 
rownames(tbl) <- c('fact.No', 'fact.Yes') 
colnames(tbl) <- c('predict.No', 'predict.Yes') 
# вектор вероятностей для перебора 
p.vector <- seq(0, 1, length = 501) 
# цикл по вероятностям отсечения 
for (p in p.vector){ 
  # прогноз 
  Прогноз <- factor(ifelse(p.qda$posterior[, 'pos'] > p, 
                           2, 1), 
                    levels = c(1, 2), 
                    labels = c('neg', 'pos')) 
  
  # фрейм со сравнением факта и прогноза 
  df.compare <- data.frame(Факт1 = Факт1, Прогноз = Прогноз) 
  
  # заполняем матрицу неточностей 
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'neg', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'pos', ]) 
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'pos', ]) 
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'neg', ]) 
  
  # считаем характеристики 
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1]) 
  y <- c(y, TPR) 
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2]) 
  x <- c(x, 1 - SPC) 
} 
for (p in p.vector){ 
  # прогноз 
  Прогноз1 <- factor(ifelse(p.logit > p, 
                            2, 1), 
                     levels = c(1, 2), 
                     labels = c('neg', 'pos')) 
  
  # фрейм со сравнением факта и прогноза 
  df.compare <- data.frame(Факт1 = Факт1, Прогноз1 = Прогноз1) 
  
  #заполняем матрицу неточностей 
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'neg', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'pos', ]) 
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'pos', ]) 
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'neg', ]) 
  
  # считаем характеристики 
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1]) 
  y1 <- c(y1, TPR) 
  SPC <- tbl[1, 1] /
    sum(tbl[1, 1] + tbl[1, 2]) 
  x1 <- c(x1, 1 - SPC) 
} 
# строим ROC-кривую
 

par(mar = c(5, 5, 1, 1)) 
# кривая 
plot(x, y, type = 'l', col = 'blue', lwd = 3, #qda 
     xlab = '(1 - SPC)', ylab = 'TPR', 
     xlim = c(0, 1), ylim = c(0, 1)) 
lines(x1, y1, type = 'l', col = 'red', lwd = 3, #logit 
      xlab = '(1 - SPC)', ylab = 'TPR', 
      xlim = c(0, 1), ylim = c(0, 1)) 
# прямая случайного классификатора 
abline(a = 0, b = 1, lty = 3, lwd = 2) 

# точка для вероятности 0.5 
points(x[p.vector == 0.5], y[p.vector == 0.5], pch = 16) 
text(x[p.vector == 0.5], y[p.vector == 0.5], 'p = 0.5(qda)', pos = 4) 
# точка для вероятности 0.2 
points(x[p.vector == 0.2], y[p.vector == 0.2], pch = 16) 
text(x[p.vector == 0.2], y[p.vector == 0.2], 'p = 0.2(qda)', pos = 4) 

# точка для вероятности 0.5 
points(x1[p.vector == 0.5], y1[p.vector == 0.5], pch = 16) 
text(x1[p.vector == 0.5], y1[p.vector == 0.5], 'p = 0.5(logit)', pos = 4) 
# точка для вероятности 0.2 
points(x1[p.vector == 0.2], y1[p.vector == 0.2], pch = 16) 
text(x1[p.vector == 0.2], y1[p.vector == 0.2], 'p = 0.2(logit)', pos = 4)


#Сравнивая модели по Roc-кривым на обучающей выборке можно сделать вывод, что модель QDA в целом справляется лучше с задачей выявления диабета, чем модель логистической регрессии. 
#При уменьшении с 0.5 до 0.2 вероятности у обеих моделей увеличивается чувствительность, но так же растет и доля ложных срабатываний. 
#Причем доля ложных срабатываний у модели логистической регрессии больше, чем у модели QDA. 

#Сравнивая модели по Roc-кривым на тестовой выборке, можно сделать аналогичные выводы. 

#Модель QDA на исходном наборе данных работает лучше, т.к. квадратный дискриминантный анализ является более гибким методом, по сравнению с логистической регрессией. 

