# Генерируем данные ------------------------------------------------------------
library('mlbench')
library('class')
library('car')
library('class')
library('e1071')
library('MASS')

# Данные примера 3 .............................................................
# ядро
my.seed <- 12345
n <-60               # наблюдений всего
train.percent <- 0.85  # доля обучающей выборки
# x-ы -- двумерные нормальные случайные величины
set.seed(my.seed)
class.0 <- mvrnorm(45, mu = c(22, 7), 
                   Sigma = matrix(c(7.5, 0, 0, 9.3), 2, 2, byrow = T))
set.seed(my.seed + 1)
class.1 <- mvrnorm(65, mu = c(19, 17), 
                   Sigma = matrix(c(6.2, 0, 0, 17), 2, 2, byrow = T))
# записываем x-ы в единые векторы (объединяем классы 0 и 1)
x1 <- c(class.0[, 1], class.1[, 1])
x2 <- c(class.0[, 2], class.1[, 2])
# фактические классы Y
y <- c(rep(0, nrow(class.0)), rep(1, nrow(class.1)))
# классы для наблюдений сетки
rules <- function(x1, x2){
  ifelse(x2 < 1.6*x1 + 19, 0, 1)
}
# Конец данных примера 3 .......................................................
# Отбираем наблюдения в обучающую выборку --------------------------------------
set.seed(my.seed)
inTrain <- sample(seq_along(x1), train.percent*n)
x1.train <- x1[inTrain]
x2.train <- x2[inTrain]
x1.test <- x1[-inTrain]
x2.test <- x2[-inTrain]
# используем истинные правила, чтобы присвоить фактические классы
y.train <- y[inTrain]
y.test <- y[-inTrain]
# фрейм с обучающей выборкой
df.train.1 <- data.frame(x1 = x1.train, x2 = x2.train, y = y.train)
# фрейм с тестовой выборкой
df.test.1 <- data.frame(x1 = x1.test, x2 = x2.test)

png(filename = 'Обучающая выборка, факт.png', bg = 'transparent')


# Рисуем обучающую выборку графике ---------------------------------------------
# для сетки (истинных областей классов): целочисленные значения x1, x2
x1.grid <- rep(seq(floor(min(x1)), ceiling(max(x1)), by = 1),
               ceiling(max(x2)) - floor(min(x2)) + 1)
x2.grid <- rep(seq(floor(min(x2)), ceiling(max(x2)), by = 1),
               each = ceiling(max(x1)) - floor(min(x1)) + 1)
# классы для наблюдений сетки
y.grid <- rules(x1.grid, x2.grid)
# фрейм для сетки
df.grid.1 <- data.frame(x1 = x1.grid, x2 = x2.grid, y = y.grid)
# цвета для графиков
cls <- c('blue', 'orange')
cls.t <- c(rgb(0, 0, 1, alpha = 0.5), rgb(1,0.5,0, alpha = 0.5))
# график истинных классов
plot(df.grid.1$x1, df.grid.1$x2, 
     pch = '·', col = cls[df.grid.1[, 'y'] + 1],
     xlab = 'X1', ylab = 'Y1',
     main = 'Обучающая выборка, факт')
# точки фактических наблюдений
points(df.train.1$x1, df.train.1$x2,
       pch = 21, bg = cls.t[df.train.1[, 'y'] + 1], 
       col = cls.t[df.train.1[, 'y'] + 1])
dev.off()


# Байесовский классификатор ----------------------------------------------------
#  наивный байес: непрерывные объясняющие переменные
# строим модель
nb <- naiveBayes(y ~ ., data = df.train.1)
# получаем модельные значения на обучающей выборке как классы
y.nb.train <- ifelse(predict(nb, df.train.1[, -3], 
                             type = "raw")[, 2] > 0.5, 1, 0)


png(filename = 'Модель naiveBayes.png', bg = 'transparent')

# график истинных классов
plot(df.grid.1$x1, df.grid.1$x2, 
     pch = '·',  col = cls[df.grid.1[, 'y'] + 1], 
     xlab = 'X1', ylab = 'Y1',
     main = 'Обучающая выборка, модель naiveBayes')
# точки наблюдений, предсказанных по модели
points(df.train.1$x1, df.train.1$x2, 
       pch = 21, bg = cls.t[y.nb.train + 1], 
       col = cls.t[y.nb.train + 1])
dev.off()
# матрица неточностей на обучающей выборке
tbl1 <- table(y.train, y.nb.train)
tbl1

# точность, или верность (Accuracy)
Acc1 <- sum(diag(tbl1)) / sum(tbl1)
Acc1

# прогноз на тестовую выборку
y.nb.test <- ifelse(predict(nb, df.test.1, type = "raw")[, 2] > 0.5, 1, 0)
# матрица неточностей на тестовой выборке
tbl2 <- table(y.test, y.nb.test)
tbl2

# точность, или верность (Accuracy)
Acc2 <- sum(diag(tbl2)) / sum(tbl2)
Acc2

# Метод kNN --------------------------------------------------------------------
#  k = 3
# строим модель и делаем прогноз
y.knn.train <- knn(train = scale(df.train.1[, -3]), 
                   test = scale(df.train.1[, -3]),
                   cl = df.train.1$y, k = 3)

png(filename = 'Модель KNN.png', bg = 'transparent')

# график истинных классов
plot(df.grid.1$x1, df.grid.1$x2, 
     pch = '·', col = cls[df.grid.1[, 'y'] + 1],
     xlab = 'X1', ylab = 'Y1',
     main = 'Обучающая выборка, модель kNN')
# точки наблюдений, предсказанных по модели
points(df.train.1$x1, df.train.1$x2, 
       pch = 21, bg = cls.t[as.numeric(y.knn.train)], 
       col = cls.t[as.numeric(y.knn.train)])
dev.off()
# матрица неточностей на обучающей выборке
tbl3 <- table(y.train, y.knn.train)
tbl3

# точность (Accuracy)
Acc3 <- sum(diag(tbl3)) / sum(tbl3)
Acc3

# прогноз на тестовую выборку
y.knn.test <- knn(train = scale(df.train.1[, -3]), 
                  test = scale(df.test.1[, -3]),
                  cl = df.train.1$y, k = 3)
# матрица неточностей на тестовой выборке
tbl4 <- table(y.test, y.knn.test)
tbl4

# точность (Accuracy)
Acc4 <- sum(diag(tbl4)) / sum(tbl4)
Acc4


TPR=tbl2[4]/(tbl2[4]+tbl2[2])

SPC=tbl2[1]/(tbl2[3]+tbl2[1])

PPV=tbl2[4]/(tbl2[4]+tbl2[3])

NPV=tbl2[1]/(tbl2[2]+tbl2[1])

FNR=1-TPR

FPR=1-SPC

FDR=1-PPV

MCC=(tbl2[1]*tbl2[4]-tbl2[2]*tbl2[3])/(
  ((tbl2[4]+tbl2[3])*(tbl2[4]+tbl2[2])*
     (tbl2[1]+tbl2[2])*(tbl2[1]+tbl2[3]))^(1/2))

