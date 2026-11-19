# ==============================================================================
# UN MILLÓN DE DÓLARES CONTRA EL ABANDONO ESCOLAR (CHILE, 2022)
# ==============================================================================
#
# PROBLEMA
#   El Ministerio de Educación de Chile dispone de US$ 1.000.000 para acompañar
#   (tutorías, becas, seguimiento) a los alumnos en riesgo de abandonar la
#   escuela. Hay que decidir A QUIÉN darle y CUÁNTO, entre los alumnos de la
#   foto 2021, para evitar la mayor cantidad posible de abandonos en 2022.
#
# REGLA DE RENDIMIENTO
#   Invertir m dólares en un alumno reduce su riesgo de abandonar en
#       efecto(m) = 0,6 * (1 - exp(-m / 200))
#   Los primeros dólares rinden más que los últimos (rendimientos decrecientes)
#   y ningún monto baja el riesgo más del 60%.
#   Se corrige con: abandonos evitados = suma de efecto(m) sobre los alumnos que
#   recibieron plata y efectivamente abandonaron.
#
# ENFOQUE
#   1) Predecir: estimar para cada alumno de 2021 la probabilidad p de abandonar
#      en 2022 (LightGBM, con rasgos de la trayectoria del alumno).
#   2) Validar sin mirar el futuro: entrenar con años <= 2019 y evaluar en 2020.
#   3) Optimizar: como el abandono evitado esperado de un alumno es
#      p * efecto(m), repartir el presupuesto para maximizar la suma de esos
#      valores. La condición de Lagrange da
#          m = max(0, 200 * ln(p / p_corte))
#      con p_corte elegido para gastar el presupuesto.
#
# ENTRADAS 
#   chile_2014_2020.csv.gz   fotos 2014-2020 con la clase abandono_t1
#   chile_2021.csv.gz        foto 2021, sin la clase (alumnos a los que se reparte)
#
# ==============================================================================

library(dplyr)
library(readr)
library(lightgbm)
library(ggplot2)
library(gridExtra)

# ------------------------------------------------------------------------------
# PARÁMETROS DEL PROBLEMA
# ------------------------------------------------------------------------------
PRESUPUESTO  <- 1000000   # US$ disponibles
PISO_ENTREGA <- 50        # monto mínimo (US$) para quien recibe algo en la entrega
# (debe ser uno de los pisos evaluados: 0, 50 o 100)

# ------------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------------------------
# Formato numérico con punto de miles y coma decimal (p. ej. 999.973,43)
fmt <- function(x, d = 0) {
  format(round(x, d), big.mark = ".", decimal.mark = ",", nsmall = d,
         scientific = FALSE, trim = TRUE)
}

# Encabezado de cada paso en la consola
paso <- function(txt) {
  cat("\n", strrep("=", 72), "\n", txt, "\n", strrep("=", 72), "\n", sep = "")
}

# Área bajo la curva ROC (probabilidad de que un alumno que abandona tenga
# mayor p que uno que no abandona; 0,5 = azar, 1 = perfecto)
auc_roc <- function(y, p) {
  r  <- rank(p)
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

cat(strrep("=", 72), "\n")
cat("UN MILLÓN DE DÓLARES CONTRA EL ABANDONO ESCOLAR (CHILE, 2022)\n")
cat(strrep("=", 72), "\n")
cat("Problema : repartir US$", fmt(PRESUPUESTO), "entre los alumnos de la foto 2021\n")
cat("           para evitar la mayor cantidad posible de abandonos en 2022.\n")
cat("Regla    : efecto(m) = 0,6 * (1 - exp(-m/200)); rinde más cada dólar inicial.\n")
cat("Método   : predecir el riesgo p de cada alumno y repartir el presupuesto\n")
cat("           para maximizar la suma de p * efecto(m).\n")

# ==============================================================================
# PASO 1 | CARGA Y CONTROL DE LOS DATOS
# ------------------------------------------------------------------------------
# Los identificadores se leen como texto (son códigos del tipo A000123 / E01234).
# Antes de modelar se controlan dimensiones, duplicados, faltantes y la clase.
# ==============================================================================
paso("PASO 1 | CARGA Y CONTROL DE LOS DATOS")

train_df <- read_csv("chile_2014_2020.csv.gz", col_types = cols(
  id_alumno = col_character(),
  id_escuela = col_character(),
  id_escuela_origen = col_character()
))

test_df <- read_csv("chile_2021.csv.gz", col_types = cols(
  id_alumno = col_character(),
  id_escuela = col_character(),
  id_escuela_origen = col_character()
))

cat("Entrenamiento:", fmt(nrow(train_df)), "filas x", ncol(train_df), "columnas | años",
    min(train_df$agno), "a", max(train_df$agno), "\n")
cat("Foto 2021    :", fmt(nrow(test_df)), "filas x", ncol(test_df), "columnas | años",
    min(test_df$agno), "a", max(test_df$agno), "\n")

# Claves: un alumno aparece una vez por año; la foto 2021 no trae la clase
dup_train <- nrow(train_df) - nrow(distinct(train_df, id_alumno, agno))
dup_test  <- nrow(test_df) - n_distinct(test_df$id_alumno)
cat("Duplicados (id_alumno, año) en entrenamiento:", dup_train, "\n")
cat("Duplicados de id_alumno en la foto 2021     :", dup_test, "\n")
cat("Alumnos de 2021 con historial en 2014-2020  :",
    fmt(100 * mean(test_df$id_alumno %in% train_df$id_alumno), 1), "%\n")

# Valores faltantes
na_train <- colSums(is.na(train_df))
cat("Columnas con faltantes en entrenamiento:", sum(na_train > 0), "\n")
if (any(na_train > 0)) print(na_train[na_train > 0])

# La clase (abandono_t1) debe estar completa y ser 0/1 en entrenamiento, y no
# debe existir en la foto 2021
stopifnot(
  !anyNA(train_df$abandono_t1),
  all(train_df$abandono_t1 %in% c(0, 1)),
  !("abandono_t1" %in% names(test_df)),
  dup_train == 0,
  dup_test == 0
)

# La clase abandono_t1 de cada año mira el año siguiente
cat("\nTasa de abandono observada por año (% de alumnos):\n")
print(round(100 * tapply(train_df$abandono_t1, train_df$agno, mean), 2))

# ==============================================================================
# PASO 2 | RASGOS DE LA TRAYECTORIA DE CADA ALUMNO
# ------------------------------------------------------------------------------
# El mismo id_alumno identifica al alumno en todos sus años; se usa para
# construir variables que describen su evolución:
#   - variación interanual de asistencia y de promedio,
#   - años registrados en el sistema,
#   - repitencias previas acumuladas.
# Se trabaja con 2014-2021 en una sola tabla para que los rasgos de 2021 usen el
# historial anterior; las etiquetas de 2021 no existen y no intervienen.
# ==============================================================================
paso("PASO 2 | RASGOS DE LA TRAYECTORIA DE CADA ALUMNO")

full_df <- bind_rows(
  train_df %>% mutate(is_test = 0),
  test_df %>% mutate(is_test = 1)
) %>%
  arrange(id_alumno, agno) %>%
  group_by(id_alumno) %>%
  mutate(
    asistencia_lag1 = lag(asistencia, 1),
    diff_asistencia = asistencia - asistencia_lag1,
    prom_lag1 = lag(prom_gral, 1),
    diff_prom = prom_gral - prom_lag1,
    anios_registrados = row_number(),
    repitencias_previas = cumsum(lag(as.integer(sit_fin == "R"), default = 0L))
  ) %>%
  ungroup()

# LightGBM necesita las variables categóricas como enteros
cat_cols <- c(
  "cod_reg_rbd", "cod_com_rbd", "cod_depe", "cod_depe2", "rural_rbd",
  "estado_estab", "cod_ense", "cod_ense2", "cod_grado", "cod_jor",
  "cod_tip_cur", "cod_rama", "cod_espe", "gen_alu", "cod_reg_alu",
  "cod_com_alu", "sit_fin"
)

full_df <- full_df %>%
  mutate(across(all_of(cat_cols), ~ as.integer(as.factor(.x))))

train_feat <- full_df %>% filter(is_test == 0)
test_feat  <- full_df %>% filter(is_test == 1)

# Se liberan las tablas intermedias que ya no se usan (test_df se conserva para
# controlar la entrega al final)
rm(train_df, full_df)
invisible(gc())

# Variables del modelo: todas menos identificadores, año, clase y marca de prueba
exclude_cols <- c("id_alumno", "id_escuela", "id_escuela_origen", "agno", "abandono_t1", "is_test")
features <- setdiff(names(train_feat), exclude_cols)

cat("Variables del modelo:", length(features), "(de las cuales", length(cat_cols),
    "categóricas)\n")
cat("Filas de entrenamiento:", fmt(nrow(train_feat)),
    "| filas de 2021 a predecir:", fmt(nrow(test_feat)), "\n")

# ==============================================================================
# PASO 3 | VALIDACIÓN TEMPORAL
# ------------------------------------------------------------------------------
# Se entrena con fotos <= 2019 (cuya clase mira hasta 2020) y se evalúa en la foto
# 2020 (cuya clase mira 2021). Así se imita el uso real: aprender del pasado y
# predecir el año siguiente, sin mezclar años al azar. El número de iteraciones
# se elige con parada temprana sobre 2020.
# ==============================================================================
paso("PASO 3 | VALIDACIÓN TEMPORAL (ENTRENAR <= 2019, EVALUAR 2020)")

tr_idx <- train_feat$agno < 2020
va_idx <- train_feat$agno == 2020

cat("Entrenamiento (2014-2019):", fmt(sum(tr_idx)), "filas | tasa de abandono",
    fmt(100 * mean(train_feat$abandono_t1[tr_idx]), 2), "%\n")
cat("Validación (2020)        :", fmt(sum(va_idx)), "filas | tasa de abandono",
    fmt(100 * mean(train_feat$abandono_t1[va_idx]), 2), "%\n\n")

dtrain_val <- lgb.Dataset(
  data = as.matrix(train_feat[tr_idx, features]),
  label = train_feat$abandono_t1[tr_idx],
  categorical_feature = cat_cols
)

dval <- lgb.Dataset.create.valid(
  dataset = dtrain_val,
  data = as.matrix(train_feat[va_idx, features]),
  label = train_feat$abandono_t1[va_idx]
)

params <- list(
  objective = "binary",
  metric = "binary_logloss",
  learning_rate = 0.03,
  num_leaves = 31,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 1,
  seed = 42,
  verbose = -1
)

cat("Entrenamiento con parada temprana: se agregan árboles (iteraciones) mientras mejora\n")
cat("el log-loss en 2020; si no mejora durante 50 iteraciones seguidas, se detiene.\n")
val_model <- lgb.train(
  params = params,
  data = dtrain_val,
  nrounds = 1000,
  valids = list(val = dval),
  early_stopping_rounds = 50,
  verbose = -1
)

best_iter <- val_model$best_iter

# Evolución del log-loss en 2020 durante el entrenamiento
curva <- unlist(val_model$record_evals$val$binary_logloss$eval)
if (length(curva) > 0) {
  marcas <- sort(unique(c(1, seq(50, length(curva), by = 50), best_iter, length(curva))))
  cat("\nIteraciones entrenadas:", length(curva),
      "| la mejor fue la", best_iter, "(se detuvo tras 50 sin mejorar)\n")
  cat("Log-loss en 2020 por iteración:\n")
  for (i in marcas) cat(sprintf("  iteración %3d: %s\n", i, fmt(curva[i], 5)))
}

# Calidad de las probabilidades en 2020
pv   <- predict(val_model, as.matrix(train_feat[va_idx, features]))
y_va <- train_feat$abandono_t1[va_idx]
brier_modelo <- mean((pv - y_va)^2)
brier_ref    <- mean(y_va) * (1 - mean(y_va))   # predictor constante = tasa real

cat("\nResultados en 2020 (año no usado para entrenar):\n")
cat(" - Iteraciones elegidas:", best_iter, "\n")
cat(" - Log-loss:", fmt(val_model$best_score, 5), "\n")
cat(" - Brier:", fmt(brier_modelo, 5), "| referencia (predictor constante):",
    fmt(brier_ref, 5), "\n")
cat(" - AUC (ROC):", fmt(auc_roc(y_va, pv), 3), "\n")
cat(" - Tasa real de abandono:", fmt(100 * mean(y_va), 2), "% | probabilidad media predicha:",
    fmt(100 * mean(pv), 2), "%\n")
cat("   Lo que importa para repartir es el orden y las razones entre las p de los\n")
cat("   alumnos; un nivel general algo distinto no cambia el reparto.\n")

# ==============================================================================
# PASO 4 | MODELO FINAL Y PREDICCIÓN PARA LA FOTO 2021
# ------------------------------------------------------------------------------
# Se reentrena con todas las fotos 2014-2020 (más datos y el año más cercano) y
# el número de iteraciones elegido en la validación, y se predice la probabilidad
# de abandonar en 2022 para cada alumno de la foto 2021.
# ==============================================================================
paso("PASO 4 | MODELO FINAL Y PREDICCIÓN PARA 2021")

dtrain_full <- lgb.Dataset(
  data = as.matrix(train_feat[, features]),
  label = train_feat$abandono_t1,
  categorical_feature = cat_cols
)

final_model <- lgb.train(
  params = params,
  data = dtrain_full,
  nrounds = best_iter,
  verbose = -1
)

p_2021 <- predict(final_model, as.matrix(test_feat[, features]))

cat("Modelo reentrenado con", fmt(nrow(train_feat)), "filas y", best_iter, "iteraciones.\n")
cat("Probabilidad de abandono predicha para", fmt(length(p_2021)), "alumnos de 2021:\n")
cat(" - media:", fmt(100 * mean(p_2021), 2), "% | mediana:", fmt(100 * median(p_2021), 2),
    "% | máxima:", fmt(100 * max(p_2021), 1), "%\n")

# ==============================================================================
# PASO 5 | REPARTO ÓPTIMO DEL PRESUPUESTO
# ------------------------------------------------------------------------------
# Objetivo: maximizar la suma sobre alumnos de  p_i * 0,6 * (1 - exp(-m_i/200))
# sujeto a  suma(m_i) <= PRESUPUESTO  y  m_i >= 0.
# Condición de optimalidad (Lagrange): el rendimiento marginal de cada dólar se
# iguala entre los alumnos que reciben plata, lo que da
#       m_i = max(0, 200 * ln(p_i / p_corte)).
# Recibe plata solo quien tiene p >= p_corte, y recibe más cuanto mayor es p
# respecto de p_corte. p_corte se busca numéricamente (uniroot) para que el
# gasto total alcance el presupuesto.
#
# Piso mínimo (m_min): decisión operativa que exige que quien recibe algo reciba
# al menos m_min dólares, para evitar becas simbólicas. Con m_min = 0 se obtiene
# el óptimo teórico. Los montos se truncan a centavos; por la forma de la regla y
# el truncado, el gasto puede quedar levemente por debajo del presupuesto.
# ==============================================================================
paso("PASO 5 | REPARTO ÓPTIMO DEL PRESUPUESTO Y COMPARACIÓN DE PISOS")

asignar_presupuesto_con_piso <- function(probs, presupuesto = PRESUPUESTO, m_min = 50) {
  probs_safe <- pmax(probs, 1e-12)
  
  # Gasto total (menos el presupuesto) si se elige p_corte como umbral
  exceso_gasto <- function(p_corte) {
    seleccionados <- probs_safe >= p_corte
    if (!any(seleccionados)) return(-presupuesto)
    
    montos <- pmax(m_min, 200 * log(probs_safe[seleccionados] / p_corte))
    sum(montos) - presupuesto
  }
  
  # Montos (truncados a centavos) que corresponden a un umbral p_corte
  montos_para <- function(p_corte) {
    montos <- ifelse(
      probs_safe >= p_corte,
      pmax(m_min, 200 * log(probs_safe / p_corte)),
      0
    )
    floor(montos * 100) / 100
  }
  
  sol <- uniroot(exceso_gasto, interval = c(1e-7, max(probs_safe) - 1e-9), tol = 1e-9)
  p_corte_opt <- sol$root
  montos_opt <- montos_para(p_corte_opt)
  
  # Garantía de presupuesto: con piso, el gasto salta en escalones y la raíz puede
  # quedar apenas del lado que lo supera; en ese caso se sube el umbral hasta que
  # el gasto no exceda el presupuesto.
  while (sum(montos_opt) > presupuesto) {
    p_corte_opt <- p_corte_opt * 1.0001
    montos_opt <- montos_para(p_corte_opt)
  }
  
  list(montos = montos_opt, p_corte = p_corte_opt)
}

# Abandonos evitados esperados según el modelo: suma de p * efecto(m)
valor <- function(m) sum(p_2021 * 0.6 * (1 - exp(-m / 200)))

pisos <- c(0, 50, 100)
resultados <- setNames(
  lapply(pisos, function(m) asignar_presupuesto_con_piso(p_2021, PRESUPUESTO, m)),
  as.character(pisos)
)
res_0   <- resultados[["0"]]
res_50  <- resultados[["50"]]
res_100 <- resultados[["100"]]

describir <- function(titulo, res, ref = NULL) {
  cat(titulo, "\n")
  cat(sprintf(" - Alumnos beneficiados: %s (%s%% de los alumnos)\n",
              fmt(sum(res$montos > 0)), fmt(100 * mean(res$montos > 0), 2)))
  cat(" - Monto máximo: US$", fmt(max(res$montos), 2), "\n")
  cat(" - Gasto total: US$", fmt(sum(res$montos), 2), "\n")
  cat(" - Abandonos evitados esperados:", fmt(valor(res$montos), 1), "\n")
  if (!is.null(ref)) {
    cat(" - Pérdida de eficiencia respecto del óptimo sin piso:",
        fmt(100 * (1 - valor(res$montos) / valor(ref$montos)), 2), "%\n")
  }
  cat("\n")
}

describir("Sin piso (óptimo teórico):", res_0)
describir("Piso de US$ 50:", res_50, res_0)
describir("Piso de US$ 100:", res_100, res_0)

# ==============================================================================
# PASO 6 | ENTREGA Y CONTROLES
# ------------------------------------------------------------------------------
# Condiciones de la consigna: solo alumnos de la foto 2021, una fila por alumno,
# montos >= 0 y suma que no supere el presupuesto.
# ==============================================================================
paso("PASO 6 | ENTREGA Y CONTROLES")

res_final <- resultados[[as.character(PISO_ENTREGA)]]

entrega <- tibble(
  id_alumno = test_feat$id_alumno,
  monto = res_final$montos
)

controles <- c(
  "Una fila por alumno de la foto 2021"       = nrow(entrega) == nrow(test_df),
  "Todos los id_alumno pertenecen a 2021"     = all(entrega$id_alumno %in% test_df$id_alumno),
  "Sin id_alumno duplicados"                  = !anyDuplicated(entrega$id_alumno),
  "Sin montos faltantes"                      = !anyNA(entrega$monto),
  "Montos mayores o iguales a cero"           = all(entrega$monto >= 0),
  "Suma de montos no supera el presupuesto"   = sum(entrega$monto) <= PRESUPUESTO
)
cat("Controles de la entrega:\n")
for (nm in names(controles)) {
  cat(sprintf("  [%s] %s\n", ifelse(controles[[nm]], "OK", "FALLA"), nm))
}
stopifnot(all(controles))

write_csv(entrega, "entrega_final.csv")

beneficiarios <- entrega$monto > 0
cat("\nEntrega con piso de US$", PISO_ENTREGA, "guardada en 'entrega_final.csv':\n")
cat(" - Filas:", fmt(nrow(entrega)), "| columnas: id_alumno, monto\n")
cat(" - Beneficiados:", fmt(sum(beneficiarios)), "| gasto total: US$",
    fmt(sum(entrega$monto), 2), "| sin gastar: US$", fmt(PRESUPUESTO - sum(entrega$monto), 2), "\n")
cat(" - Monto entre beneficiados: mínimo US$", fmt(min(entrega$monto[beneficiarios]), 2),
    "| mediana US$", fmt(median(entrega$monto[beneficiarios]), 2),
    "| promedio US$", fmt(mean(entrega$monto[beneficiarios]), 2),
    "| máximo US$", fmt(max(entrega$monto), 2), "\n")
cat(" - Riesgo medio predicho: beneficiados", fmt(100 * mean(p_2021[beneficiarios]), 2),
    "% frente a", fmt(100 * mean(p_2021), 2), "% del total de alumnos\n")
cat(" - Probabilidad de corte (p_corte):", fmt(100 * res_final$p_corte, 3), "%\n")

# ==============================================================================
# PASO 7 | GRÁFICOS
# ------------------------------------------------------------------------------
# Distribución de los montos y cobertura para los pisos de US$ 50 y de US$ 100,
# y resumen de sensibilidad (beneficiados y monto máximo según el piso).
# ==============================================================================
paso("PASO 7 | GRÁFICOS")

total_alumnos <- nrow(entrega)

# --- Piso de US$ 50 ---
beneficiados_50 <- res_50$montos[res_50$montos > 0]
total_beneficiados_50 <- length(beneficiados_50)
total_sin_beca_50 <- total_alumnos - total_beneficiados_50

p1_50 <- ggplot(data.frame(monto = beneficiados_50), aes(x = monto)) +
  geom_histogram(bins = 30, fill = "#1f77b4", color = "white", alpha = 0.85) +
  geom_vline(aes(xintercept = mean(beneficiados_50)), color = "#d62728", linetype = "dashed", linewidth = 1.2) +
  geom_vline(aes(xintercept = median(beneficiados_50)), color = "#2ca02c", linetype = "solid", linewidth = 1.2) +
  labs(
    title = "Distribución (Piso Mínimo US$ 50)",
    subtitle = "Línea roja: promedio | línea verde: mediana",
    x = "Monto en US$",
    y = "Cantidad de Alumnos"
  ) +
  theme_minimal()

df_cobertura_50 <- data.frame(
  Categoria = c("Sin Asignación ($0)", "Con Beca (>= $50)"),
  Cantidad = c(total_sin_beca_50, total_beneficiados_50)
)

p2_50 <- ggplot(df_cobertura_50, aes(x = Categoria, y = Cantidad, fill = Categoria)) +
  geom_bar(stat = "identity", width = 0.5, color = "black", linewidth = 0.3, alpha = 0.85) +
  geom_text(aes(label = fmt(Cantidad)), vjust = -0.8, fontface = "bold", size = 4) +
  labs(
    title = paste0("Cobertura Total (", fmt(total_alumnos), " alumnos)"),
    x = NULL,
    y = "Número de Alumnos"
  ) +
  scale_y_continuous(labels = function(x) fmt(x)) +
  scale_fill_manual(values = c("Sin Asignación ($0)" = "#aec7e8", "Con Beca (>= $50)" = "#1f77b4")) +
  theme_minimal() +
  theme(legend.position = "none")

grafico_final_50 <- grid.arrange(p1_50, p2_50, ncol = 2)
ggsave("comparacion_asignacion.png", grafico_final_50, width = 14, height = 6, dpi = 300)
cat("Gráfico guardado: comparacion_asignacion.png\n")

# --- Piso de US$ 100 ---
beneficiados_100 <- res_100$montos[res_100$montos > 0]
total_beneficiados_100 <- length(beneficiados_100)
total_sin_beca_100 <- total_alumnos - total_beneficiados_100

p1_100 <- ggplot(data.frame(monto = beneficiados_100), aes(x = monto)) +
  geom_histogram(bins = 30, fill = "#2ca02c", color = "white", alpha = 0.85) +
  geom_vline(aes(xintercept = mean(beneficiados_100)), color = "#d62728", linetype = "dashed", linewidth = 1.2) +
  geom_vline(aes(xintercept = median(beneficiados_100)), color = "#1f77b4", linetype = "solid", linewidth = 1.2) +
  labs(
    title = "Distribución (Piso Mínimo US$ 100)",
    subtitle = "Línea roja: promedio | línea azul: mediana",
    x = "Monto en US$",
    y = "Cantidad de Alumnos"
  ) +
  theme_minimal()

df_cobertura_100 <- data.frame(
  Categoria = c("Sin Asignación ($0)", "Con Beca (>= $100)"),
  Cantidad = c(total_sin_beca_100, total_beneficiados_100)
)

p2_100 <- ggplot(df_cobertura_100, aes(x = Categoria, y = Cantidad, fill = Categoria)) +
  geom_bar(stat = "identity", width = 0.5, color = "black", linewidth = 0.3, alpha = 0.85) +
  geom_text(aes(label = fmt(Cantidad)), vjust = -0.8, fontface = "bold", size = 4) +
  labs(
    title = paste0("Cobertura Total (", fmt(total_alumnos), " alumnos)"),
    x = NULL,
    y = "Número de Alumnos"
  ) +
  scale_y_continuous(labels = function(x) fmt(x)) +
  scale_fill_manual(values = c("Sin Asignación ($0)" = "#c7e9c0", "Con Beca (>= $100)" = "#2ca02c")) +
  theme_minimal() +
  theme(legend.position = "none")

grafico_final_100 <- grid.arrange(p1_100, p2_100, ncol = 2)
ggsave("comparacion_asignacion_100.png", grafico_final_100, width = 14, height = 6, dpi = 300)
cat("Gráfico guardado: comparacion_asignacion_100.png\n")

# --- Sensibilidad al piso ---
df_sensibilidad <- data.frame(
  Piso = factor(c("Piso US$ 50", "Piso US$ 100"), levels = c("Piso US$ 50", "Piso US$ 100")),
  Beneficiados = c(sum(res_50$montos > 0), sum(res_100$montos > 0)),
  MontoMax = c(max(res_50$montos), max(res_100$montos))
)

p_ben <- ggplot(df_sensibilidad, aes(x = Piso, y = Beneficiados, fill = Piso)) +
  geom_bar(stat = "identity", width = 0.5, alpha = 0.85, color = "black", linewidth = 0.3) +
  geom_text(aes(label = fmt(Beneficiados)), vjust = -0.8, fontface = "bold", size = 4) +
  labs(title = "Alumnos Beneficiados según el Piso", x = NULL, y = "Cantidad de Alumnos") +
  scale_fill_manual(values = c("Piso US$ 50" = "#1f77b4", "Piso US$ 100" = "#2ca02c")) +
  theme_minimal() +
  theme(legend.position = "none")

p_max <- ggplot(df_sensibilidad, aes(x = Piso, y = MontoMax, fill = Piso)) +
  geom_bar(stat = "identity", width = 0.5, alpha = 0.85, color = "black", linewidth = 0.3) +
  geom_text(aes(label = paste0("US$ ", fmt(MontoMax, 2))), vjust = -0.8, fontface = "bold", size = 4) +
  labs(title = "Monto Máximo Entregado", x = NULL, y = "Monto en US$") +
  scale_fill_manual(values = c("Piso US$ 50" = "#1f77b4", "Piso US$ 100" = "#2ca02c")) +
  theme_minimal() +
  theme(legend.position = "none")

grafico_sensibilidad <- grid.arrange(p_ben, p_max, ncol = 2)
ggsave("sensibilidad_pisos.png", grafico_sensibilidad, width = 12, height = 5, dpi = 300)
cat("Gráfico guardado: sensibilidad_pisos.png\n")

# ==============================================================================
# RESUMEN Y LÍMITES DE LA SOLUCIÓN
# ==============================================================================
paso("RESUMEN")

cat("Qué se hizo:\n")
cat(" 1. Se estimó la probabilidad de abandono de cada alumno de 2021 con LightGBM,\n")
cat("    validando en 2020 sin usar información posterior.\n")
cat(sprintf(" 2. Se repartieron US$ %s con la regla m = 200*ln(p/p_corte),\n", fmt(PRESUPUESTO)))
cat(sprintf("    que maximiza los abandonos evitados esperados, con un piso de US$ %s.\n", fmt(PISO_ENTREGA)))
cat(" 3. Se verificó que la entrega cumple las condiciones de la consigna.\n\n")

cat("Resultado: ", fmt(sum(beneficiarios)), " alumnos reciben entre US$ ",
    fmt(min(entrega$monto[beneficiarios]), 2), " y US$ ", fmt(max(entrega$monto), 2),
    "; abandonos evitados esperados: ", fmt(valor(entrega$monto), 1), ".\n", sep = "")
cat("Costo del piso respecto del óptimo sin piso: ",
    fmt(100 * (1 - valor(entrega$monto) / valor(res_0$montos)), 2), "% de los abandonos evitados esperados.\n\n", sep = "")

cat("Límites a tener en cuenta:\n")
cat(" - Las p miden riesgo de abandonar, no el efecto de la plata sobre cada alumno.\n")
cat("   El reparto supone, como indica la hipótesis del ministerio, que la plata solo\n")
cat("   rinde en quien iba a abandonar; saber si el programa funciona requiere\n")
cat("   variación creíble en quién lo recibe (experimento o diseño causal).\n")
cat(" - Los abandonos evitados esperados dependen del nivel de las p del modelo; la\n")
cat("   comparación entre pisos no depende de ese nivel.\n")
cat(" - 2020 fue un año atípico (menor abandono que 2019): la validación es una\n")
cat("   aproximación de lo que ocurra en 2022.\n\n")

cat("Archivos generados:\n")
for (f in c("entrega_final.csv", "comparacion_asignacion.png",
            "comparacion_asignacion_100.png", "sensibilidad_pisos.png")) {
  cat(sprintf("  [%s] %s\n", ifelse(file.exists(f), "OK", "NO ENCONTRADO"), f))
}

