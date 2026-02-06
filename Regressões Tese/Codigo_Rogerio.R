# Carregar o pacote readr (mais rápido e eficiente)
library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(fixest)
library(modelsummary)
library(flextable)
library(officer)

################ Dados ################
dados_completos <- read_excel(path = "C:/Users/jorge/Downloads/Dados_Rogerio_BR.xlsx", na="NA")

dados = dados_completos[,which(names(dados_completos) %in% c('ticker', 'ano', 'acoes','vm','llcor','llhist','plcor','plhist'))]

dados$puhist = dados$vm/dados$acoes
dados$epshist = dados$llhist/dados$acoes
dados$epscor = dados$llcor/dados$acoes
dados$bvpshist = dados$plhist/dados$acoes
dados$bvpscor = dados$plcor/dados$acoes


dados_completos1 <- read_excel(path = "C:/Users/jorge/Downloads/Dados_Rogerio_PT.xlsx", na="NA")

dados1 = dados_completos1[,which(names(dados_completos1) %in% c('ticker', 'ano', 'acoes','vm','llcor','llhist','plcor','plhist'))]

###### Estatítisca Descritiva ##########
dados_vm$pais = "Brasil"
dados_vm1$pais = "Portugal"
dados_desc = rbind(dados_vm, dados_vm1)

# 1) Definir o grupo: Cumpriu vs Não Cumpriu
dados2 <- dados_desc %>%
  mutate(
    grupo = if_else(pais == "Brasil",
                    "Brasil", "Portugal")
  )

# 2) Escolher as variáveis númericas

vars_numericas <- c("plhist","plcor","llhist","llcor")

# 3) Calcular estatísticas por variável e por grupo
sumstats_long <- dados2 %>%
  select(grupo, all_of(vars_numericas)) %>%
  pivot_longer(-grupo, names_to = "Variável", values_to = "valor") %>%
  group_by(Variável, grupo) %>%
  summarize(
    `Média`          = mean(valor, na.rm = TRUE),
    `Desvio-Padrão`  = sd(valor, na.rm = TRUE),
    `Mínimo`         = suppressWarnings(min(valor, na.rm = TRUE)),
    `Máximo`         = suppressWarnings(max(valor, na.rm = TRUE)),
    .groups = "drop"
  )

ttests <- dados2 %>%
  select(grupo, all_of(vars_numericas)) %>%
  pivot_longer(-grupo, names_to = "Variável", values_to = "valor") %>%
  group_by(Variável) %>%
  summarize(
    dif_media = mean(valor[grupo == "Cumpriu a Meta"], na.rm = TRUE) -
      mean(valor[grupo == "Não Cumpriu a Meta"], na.rm = TRUE),
    p_valor   = tryCatch(t.test(valor ~ grupo)$p.value, error = function(e) NA_real_),
    .groups = "drop"
  ) %>%
  mutate(
    estrelas = case_when(
      is.na(p_valor) ~ "",
      p_valor < 0.01 ~ "***",
      p_valor < 0.05 ~ "**",
      p_valor < 0.10 ~ "*",
      TRUE ~ ""
    ),
    Dif_Média = paste0(round(dif_media, 4), estrelas)  # coluna já “formatada”
  ) %>%
  select(Variável, Dif_Média)  # (se quiser p-valor também, me diga)


# 4) Abrir em formato wide, com um bloco para cada grupo
sumstats_wide <- sumstats_long %>%
  pivot_wider(
    id_cols = Variável,
    names_from = grupo,
    values_from = c(`Média`, `Desvio-Padrão`, `Mínimo`, `Máximo`)
  )

sumstats_wide <- sumstats_wide %>%
  left_join(ttests, by = "Variável")

# 5) Organizar as colunas na ordem pedida
ordem_cols <- c(
  "Variável",
  # bloco Cumpriu
  "Média_Brasil",
  "Desvio-Padrão_Brasil",
  "Mínimo_Brasil",
  "Máximo_Brasil",
  # bloco Não Cumpriu
  "Média_Portugal",
  "Desvio-Padrão_Portugal",
  "Mínimo_Portugal",
  "Máximo_Portugal",
  "Dif_Média"
)

tabela_base <- sumstats_wide %>%
  # renomear para o padrão "Métrica_Grupo"
  rename_with(.fn = ~ str_replace_all(.x, " ", " "),
              .cols = everything()) %>%
  # garantir que colunas existem antes de reordenar
  { .[, intersect(ordem_cols, c("Variável", names(.))) ] }

tabela_base <- tabela_base %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

# 6) Criar o flextable com cabeçalho em dois níveis
ft <- flextable(tabela_base)

# Linha 1 do cabeçalho (mesclagens por bloco)
ft <- add_header_row(
  ft,
  values = c("Variável", "Brasil", "Portugal", "Diferença"),
  colwidths = c(1, 4, 4, 1)
)

ft <- merge_v(ft, j = "Variável", part = "header")
ft <- compose(ft, i = 1, j = "Dif_Média", value = as_paragraph("Diferença de Médias"), part = "header")
ft <- compose(ft, i = 2, j = "Dif_Média", value = as_paragraph("Diferença de Médias"), part = "header")
ft <- merge_v(ft, j = "Dif_Média", part = "header")

vals <- c("Média","Desvio-Padrão","Mínimo", "Máximo","Média","Desvio-Padrão","Mínimo", "Máximo")
cols <- 2:9
for (k in seq_along(cols)) {
  ft <- compose(ft, i = 2, j = cols[k], value = as_paragraph(vals[k]), part = "header")
}

n_cumpriu <- dados2 %>% filter(grupo == "Brasil") %>% nrow()
n_nao     <- dados2 %>% filter(grupo == "Portugal") %>% nrow()

ft <- add_body_row(
  ft,
  values = c("N", rep(n_cumpriu, 4), rep(n_nao, 4), ""),
  top = FALSE
)
ft <- merge_h(ft, i = nrow(tabela_base) + 1, part = "body")
ft <- align(ft, align = "center", part = "all")

ft = add_footer_lines(ft, "Nota: VM corresponde ao valor de mercado. PL e RL correspondem ao patrimônio líquido e resultado líquido, respectivamente, sendo o sobreescrito 'hist' para histórico e 'cor' para corrigido.")
ft = add_footer_lines(ft, "Fonte: Elaborado pelo autor.")
ft = autofit(ft)
save_as_docx(ft,path = "ft.docx")



######### Modelos #######

###### Modelo  Brasil ##########
dados_vm = dados[,which(names(dados) %in% c('ticker','ano','vm','plhist','plcor','llhist','llcor'))]
dados_vm = na.omit(dados_vm)
mod1 = feols(vm ~ plhist | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm)
mod2 = feols(vm ~ plcor | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm)
mod3 = feols(vm ~ llhist | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm)
mod4 = feols(vm ~ llcor | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm)
mod5 = feols(vm ~ plhist+plcor+llhist+llcor | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm)


tablemods = list(
  "(1)" = mod1,
  "(2)" = mod2,
  "(3)" = mod3,
  "(4)" = mod4,
  "(5)" = mod5
)

coef_map = c('plhist'='PLhist',
             'plcor'='PLcor',
             'llhist'='RLhist',
             'llcor'='RLcor')

Tablemods = modelsummary(
  tablemods,
  stars = c('*' = .1,'**'=0.05,'***' = .01),
  coef_map = coef_map,
  output = "flextable"
)

Tablemods = add_header_lines(Tablemods, values = c("Tabela 1 - Modelo de regressão do value-relevance do lucro e PL histórico versus corrigido no Brasil"))
Tablemods = align(Tablemods, i=1, align = "left", part = "header")
Tablemods = hline_top(Tablemods, part = 'header', border = fp_border_default(width = 0))
Tablemods = compose(Tablemods,i=2,j=1,part = 'header',value = as_paragraph('Variable'))
Tablemods = add_footer_lines(Tablemods, "Nota: Variável dependente corresponde ao valor de mercado. PL e RL correspondem ao patrimônio líquido e resultado líquido, respectivamente, sendo o sobreescrito 'hist' para histórico e 'cor' para corrigido. EF corresponde ao efeito fixo. Erro-padrão clusterizado por empresa entre parênteses. *, **, *** indica o nível de significância em 10, 5 e 1%, respectivamente.")
Tablemods = add_footer_lines(Tablemods, "Fonte: Elaborado pelo autor.")
Tablemods = autofit(Tablemods)
save_as_docx(Tablemods,path = "Tablemods.docx")


AIC(mod1, mod2, mod3)[3]-AIC(mod1, mod2, mod3)[1]
BIC(mod1, mod2, mod3)

# Variance Inflation

library(car)
library(dplyr)
library(flextable)

# Modelo (se já existir, pode pular)
m <- lm(vm ~ plhist+plcor+llhist+llcor, data = dados_vm)

# VIF
v <- car::vif(m)

# Tabela
threshold <- 5  # você pode mudar para 10 se preferir

tab_vif <- tibble(
  Variável = c("PLhist", "PLcor", "RLhist","RLcor"),
  VIF      = as.numeric(v),
  Threshold = threshold,
  Status   = if_else(VIF >= threshold, "Atenção", "OK")
) %>%
  mutate(VIF = round(VIF, 4))

# Flextable
ft_vif <- flextable(tab_vif)

ft_vif <- align(ft_vif, align = "center", part = "all")
ft_vif <- align(ft_vif, j = "Variável", align = "left", part = "body")
ft_vif <- align(ft_vif, j = "Variável", align = "left", part = "header")

ft_vif <- add_footer_lines(
  ft_vif,
  "Nota: VIF abaixo de 5 sugere baixa multicolinearidade. CUMP: Distância entre a emissão e a meta proporcional ao valor emitido reescalonado para ter média igual a zero e desvio-padrão igual a 1. TAM: Logaritimo natural dos ativo total. ROA: Retorno dos ativos."
)
ft_vif <- add_footer_lines(ft_vif, "Fonte: Elaborado pelo autor.")

ft_vif <- autofit(ft_vif)

save_as_docx(ft_vif, path = "vif.docx")

###### Modelo  PT ##########
dados_vm1 = dados1[,which(names(dados1) %in% c('ticker','ano','vm','plhist','plcor','llhist','llcor'))]
dados_vm1 = na.omit(dados_vm1)
mod6 = feols(vm ~ plhist | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm1)
mod7 = feols(vm ~ plcor | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm1)
mod8 = feols(vm ~ llhist | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm1)
mod9 = feols(vm ~ llcor | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm1)
mod10 = feols(vm ~ plhist+llhist+llcor | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm1)
mod11 = feols(vm ~ plcor+llhist+llcor | ticker + ano, panel.id = ~ticker+ano, cluster = "ticker", data = dados_vm1)


tablemods1 = list(
  "(1)" = mod6,
  "(2)" = mod7,
  "(3)" = mod8,
  "(4)" = mod9,
  "(5)" = mod10,
  "(6)" = mod11
)

coef_map = c('plhist'='PLhist',
             'plcor'='PLcor',
             'llhist'='RLhist',
             'llcor'='RLcor')

Tablemods1 = modelsummary(
  tablemods1,
  stars = c('*' = .1,'**'=0.05,'***' = .01),
  coef_map = coef_map,
  output = "flextable"
)

Tablemods1 = add_header_lines(Tablemods1, values = c("Tabela 2 - Modelo de regressão do value-relevance do lucro e PL histórico versus corrigido em Portugal"))
Tablemods1 = align(Tablemods1, i=1, align = "left", part = "header")
Tablemods1 = hline_top(Tablemods1, part = 'header', border = fp_border_default(width = 0))
Tablemods1 = compose(Tablemods1,i=2,j=1,part = 'header',value = as_paragraph('Variable'))
Tablemods1 = add_footer_lines(Tablemods1, "Nota: Variável dependente corresponde ao valor de mercado. PL e RL correspondem ao patrimônio líquido e resultado líquido, respectivamente, sendo o sobreescrito 'hist' para histórico e 'cor' para corrigido. EF corresponde ao efeito fixo. Erro-padrão clusterizado por empresa entre parênteses. *, **, *** indica o nível de significância em 10, 5 e 1%, respectivamente.")
Tablemods1 = add_footer_lines(Tablemods1, "Fonte: Elaborado pelo autor.")
Tablemods1 = autofit(Tablemods1)
save_as_docx(Tablemods1,path = "Tablemods1.docx")


AIC(mod1, mod2, mod3)[3]-AIC(mod1, mod2, mod3)[1]
BIC(mod1, mod2, mod3)

# Variance Inflation

library(car)
library(dplyr)
library(flextable)

# Modelo (se já existir, pode pular)
m <- lm(vm ~ plhist+plcor+llhist+llcor, data = dados_vm1)

# VIF
v <- car::vif(m)

# Tabela
threshold <- 5  # você pode mudar para 10 se preferir

tab_vif <- tibble(
  Variável = c("PLhist", "PLcor", "RLhist","RLcor"),
  VIF      = as.numeric(v),
  Threshold = threshold,
  Status   = if_else(VIF >= threshold, "Atenção", "OK")
) %>%
  mutate(VIF = round(VIF, 4))

# Flextable
ft_vif <- flextable(tab_vif)

ft_vif <- align(ft_vif, align = "center", part = "all")
ft_vif <- align(ft_vif, j = "Variável", align = "left", part = "body")
ft_vif <- align(ft_vif, j = "Variável", align = "left", part = "header")

ft_vif <- add_footer_lines(
  ft_vif,
  "Nota: "
)
ft_vif <- add_footer_lines(ft_vif, "Fonte: Elaborado pelo autor.")

ft_vif <- autofit(ft_vif)

save_as_docx(ft_vif, path = "vif1.docx")
