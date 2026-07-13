# ==============================================================================
# IMPORTS
# ==============================================================================
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import (
    col, count, when, isnan, to_date, coalesce, to_timestamp, lit, concat,
    upper, trim, current_date, datediff
)

# ==============================================================================
# 1 - Configurar leitura dos dados
# ==============================================================================
spark = SparkSession.builder    \
    .appName("Case IAM")        \
    .master("local[*]")         \
    .getOrCreate()

# Ler arquivo parquet
df_raw = spark.read.parquet("file_name.parquet")    # Dados brutos para consultas
df = spark.read.parquet("file_name.parquet")

print(f'Registros carregados: {df.count()}')
print(f'Colunas: {len(df.columns)}')
df.printSchema()

# ==============================================================================
# 2 - Análise exploratória e diagnóstico de qualidade
# ==============================================================================
# Verificar schema original
df_raw.printSchema()

## Identificar padrões de datas
# data_abertura_conta - mistura yyy-MM-dd e dd/MM/yyyy
df_raw.select("data_abertura_conta").where(col("data_abertura_conta").isNotNull()).show(10, truncate=False)

# data_criacao_operador - mistura yyyy/MM/dd e yyy-MM-dd
df_raw.filter(
    ~col("data_criacao_operador").rlike("^\\d{4}-\\d{2}-\\d{2}$")
    & col("data_criacao_operador").isNotNull()
).select("data_criacao_operador").distinct().show(20, truncate=False)

# ref_anomes - formato yyyyMM
df_raw.filter(
    ~col("ref_anomes").rlike("^\\d{4}-\\d{2}-\\d{2}$")
    & col("ref_anomes").isNotNull()
).select("ref_anomes").distinct().show(20, truncate=False)

# Contar nulos por coluna
df_raw.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df_raw.columns
]).show(vertical=True)

# Verificar duplicatas exatas
total_bruto = df_raw.count()
print(f'Total bruto:    {total_bruto}')

distintas = df_raw.distinct().count()
print(f'Total distintas:    {distintas}')

print(f'Total duplicadas {total_bruto - distintas}')

# ==============================================================================
# 3 - Limpeza e padronização
# ==============================================================================
# Capturar total original ANTES de qualquer modificação
total_antes = df.count()
print(f'Total de registros (original)': {total_antes})

# Colunas que serão convertidas para 'int'
colunas_int = [
    "quantidade_socios",
    "cod_operador",
    "qtd_operadores_associados_ao_cnpj",
    "qtd_contas_associadas_ao_operador",
    "qtd_total_sessoes",
    "qtd_sessoes_mobile",
    "qtd_sessoes_web",
    "qtd_sessoes_app_comp",
    "qtd_total_sessoes_90d_anteriores",
    "qtd_sessoes_mobile_90d_anteriores",
    "qtd_sesoes_web_90d_anteriores",
    "qtd_sessoes_app_comp_90d_anteriores",
]

for c in colunas_int:
    df = df.withColumn(c, col(c).cast("integer"))

# Colunas com formato único yyyy-MM-dd
colunas_date = [
    "data_encerramento_conta",
    "data_fundacao_empresa",
    "data_inicio_relacionamento_banco",
    "ref_token",
]

for c in colunas_date:
    df = df.withColumn(c, to_date(col(c), "yyyy-MM-dd"))

# data_abertura_conta - yyyy-MM-dd e dd/MM/yyyy
df = df.withColumn(
    "data_abertura_conta", coalesce(
        to_date(col("data_abertura_conta"), "yyyy-MM-dd"),
        to_date(col("data_abertura_conta"), "dd/MM/yyyy"),
    )
)

# data_criacao_operador - yyyy-MM-dd e yyyy/MM/dd
df = df.withColumn("data_criacao_operador", coalesce(
    to_date(col("data_criacao_operador"), "yyyy-MM-dd"),
    to_date(col("data_criacao_operador"), "yyyy/MM/dd"),
))

# data_hora - microssegundos
df = df.withColumn(
    "data_hora",
    to_timestamp(col("data_hora"), "yyyy-MM-dd-HH.mm.ss.SSSSSS")
)

# data_partition_control_br - padrão com espaço
df = df.withColumn(
    "data_partition_control_br",
    to_timestamp(col("data_partition_control_br"), "yyyy-MM-dd-HH.mm.ss")
)

# ref_anomes - mistura yyyy-MM-dd, yyyy/MM/dd e yyyyMM
df =df.withColumn("ref_anomes", coalesce(
    to_date(col("ref_anomes"), "yyyy-MM-dd"),
    to_date(col("ref_anomes"), "yyyy/MM/dd"),
    to_date(concat(col("ref_anomes"), lit("01")) "yyyyMMdd")    # "202605" -> "20260501"
))

print('Conversão de tipos concluída.')
df.printSchema()

# Capturar total de nulos ANTES da correção
nulos_antes = {}
for c in df.columns:
    qtd = df.filter(col(c).isNull()).count()
    nulos_antes[c] = qtd

total_nulos_antes = sum(nulos_antes.values())
print(f'Total de nulos ANTES da correação: {total_nulos_antes}')

# Preencher nulos de inteiros com 0
df = df.fillna(0, subset=colunas_int)

# Preencher nulos de string com "NAO_INFORMADO"
colunas_string = [c.name for c in df.schema.fields if str(c.dataType) == "StringType()"]
df = df.fillna("NAO_INFORMADO", subset=colunas_string)

# Datas nulas: Manter como 'null', tratar individualmente

# Capturar nulos DEPOIS da correção
nulos_depois = {}
for c in df.columns:
    qtd = df.filter(col(c).isNull()).count()
    nulos_depois[c] = qtd

total_nulos_depois = sum(nulos_depois.values())
print(f'Total de nulos DEPOIS da correação: {total_nulos_depois}')
print(f'Nulos corrigidos:   {total_nulos_antes - total_nulos_depois}')

# Tratar duplicatas exatas
df = df.dropDuplicates()
total_depois = df.count()
duplicatas_removidas = total_antes - total_depois

print(f'Registros antes:        {total_antes}')
print(f'Registros depois:       {total_depois}')
print(f'Duplicatas removidas:   {duplicatas_removidas}')

# Padronizar colunas de categoria, uppercase e espaço
colunas_categoricas = [
    "grupo_segmento", "segmento_detalhado", "modelo_atendimento",
    "situacao_conta", "situacao_cadastral_receira", "uf_sigla",
    "tipo_sociedade", "tipo_operador", "sit_operador",
    "mot_bloq_operador", "flag_firmas_e_poderes",
    "tem_token_mobile_habilitado", "tem_token_embarcado_habilitado",
    "tipo_token", "status_token", "forma_alteracao_token",
    "flag_acessou_canal", "flag_acessou_mobile", "flag_acessou_web",
    "faixa_tempo_criacao_conta", "faixa_tempo_vida_empresa",
    "faixa_tempo_criacao_operador",
]

for c in colunas_categoricas:
    df = df.withColumn(c, upper(trim(col(c))))

# Substituir nulo em colunas string - 'NAO_INFORMADO'
valores_nulos = ["", "NULL", "N/A", "NAN", "NONE"]
colunas_string = [c.name for c in df.schema,fields if str(c.dataType) == "StringType()"]

for c in colunas_string:
    df = df.withColumn(c,
        when(upper(trim(col(c))).isin(valores_nulos), "NAO_INFORMADO")
        .otherwise(col(c))
    )

# Padronizar flags 'S/N'
colunas_flag = ["flag_acessou_canal", "flag_acessou_mobile", "flag_acessou_web"]

for c in colunas_flag:
    df = df.withColumn(c,
        when(upper(trim(col(c))).isin(["SIM", "S", "1"]), "SIM")
        .when(upper(trim(col(c))).isin(["NAO", "N", "0"]), "NAO")
        .otherwise("NAO_INFORMADO")
    )

print('Padronização concluída.')
print(f'Total de registros após limpeza completa    {df.count()}')

# ==============================================================================
# 4 - Transformações de negócio
# ==============================================================================
# Calcular diferença em dias
df = df.withColumn("dias_relacionamento",
    datediff(current_date(), col("data_inicio_relacionamento_banco"))
)

# Classificar em faixas
df = df.withColumn("faixa_tempo_relacionamento",
    when(col("dias_relacionamento").isNull(), "NAO_INFORMADO")
    .when(col("dias_relacionamento") <= 180, "1. Ate 6 meses")
    .when(col("dias_relacionamento") <= 365, "2. Entre 6 meses e 1 ano")
    .when(col("dias_relacionamento") <= 1095, "3. Entre 1 e 3 anos")
    .when(col("dias_relacionamento") <= 1825, "4. Entre 3 e 5 anos")
    .when(col("dias_relacionamento") <= 3650, "5. Entre 5 e 10 anos")
    .otherwise("6. Mais de 10 anos")
)

# Verificar distribuição
df.groupBy("faixa_tempo_relacionamento").count()    \
    .orderBy("faixa_tempo_relacionamento").show(truncate=False)

# Calcular score somando pontos por critério
score = (
    when(col("tem_token_mobile_habilitado") == "SIM", 3).otherwise(0) +
    when(col("tem_token_embarcado_habilitado") == "SIM", 2).otherwise(0) +
    when(col("flag_acesso_mobile") == "SIM", 2).otherwise(0) +
    when(col("flag_acesso_web") == "SIM", 1).otherwise(0) +
    when(col("flag_acesso_canal") == "SIM", 1).otherwise(0) +
    when(
        col("flag_acessou_canal") == "SIM" &
        (datediff(current_date(), col("ref_anomes")) <= 90),
        1
    ).otherwise(0) +
    when(col("qtd_total_sessoes") > 10, 1).otherwise(0)
)

df = df.withColumn("score_maturidade_digital", score)

# Classificação baseada no score
df = df.withColumn("classificacao_digital",
    when(col("score_maturidade_digital") <= 2, "Baixo")
    .when(col("score_maturidade_digital") <= 5, "Medio")
    .when(col("score_maturidade_digital") <= 8, "Alto")
    .otherwise("Avancado")
)

# Verificar distribuição
df.groupBy("faixa_tempo_relacionamento").count()    \
    .orderBy("faixa_tempo_relacionamento").show(truncate=False)

# Calcular score somando pontos por critério
score = (
    when(col("tem_token_mobile_habilitado") == "SIM", 3).otherwise(0) +
    when(col("tem_token_membarcado_habilitado") == "SIM", 2).otherwise(0) +
    when(col("flag_acessou_mobile") == "SIM", 2).otherwise(0) +
    when(col("flag_acessou_web") == "SIM", 1).otherwise(0) +
    when(col("flag_acessou_canal") == "SIM", 1).otherwise(0) +
    when(
        (col("flag_acessou_canal") == "SIM") &
        (datediff(current_date(), col("ref_anomes")) <= 90), 1).otherwise(0) +
    when(col("qtd_total_sessoes") > 10, 1).otherwise(0)
)

df = df.withColumn("score_maturidade_digital", score)

# Classificação baseada no score
df = df.withColumn("classificacao_digitais",
    when(col("score_maturidade_digital") <= 2, "Baixo")
    .when(col("score_maturidade_digital") <= 5, "Medio")
    .when(col("score_maturidade_digital") <= 8, "Alto")
    .otherwise("Avancado")
    )

# Verificar distribuição
df.groupBy("classificacao_digital").count().orderBy("classificacao_digital").show()
df.groupBy("score_maturidade_digital").count().orderBy("score_maturidade_digital").show()

# Condições de ALTO risco
cond_bloqueado = (col("sit_operador") == "BLOQUEADO")
cond_sem_token_com_poderes = (
    (col("tem_token_mobile_habilitado") == "NAO") &
    (col("tem_token_embarcado_habilitado") == "NAO") &
    (col("flag_firmas_e_poderes") == "S")
)

cond_muitas_contas = (col("qtd_contas_associadas_ao_operador") > 10)

alto = cond_bloqueado | cond_sem_token_com_poderes | cond_muitas_contas

# Condições de MÉDIO risco
cond_token_unico = (
    col("tem_token_mobile_habilitado") != col("tem_token_embarcado_habilitado")
)

cond_sem_acesso_90dias = (
    (col("flag_acessou_canal") == "NAO") |
    (datediff(current_date(), col("ref_anomes")) >= 90)
)

medio = cond_token_unico | cond_sem_acesso_90dias

# Classificar - ordem: ALTO > MÉDIO > BAIXO
df = df.withColumn("risco_operador",
    when(alto, "ALTO")
    .when(medio, "MEDIO")
    .otherwise("BAIXO")
)

# Verificar distribuição
df.groupBy("risco_operador").count().orderBy("risco_operador").show()

df = df.withColumn("concentracao_operadores",
    when(col("qtd_operadores_associados_ao_cnpj") == 1, "Operador Unico")
    .when(col("qtd_operadores_associados_ao_cnpj") <= 3, "Baixa Concentracao")
    .when(col("qtd_operadores_associados_ao_cnpj") <= 10, "Media Concentracao")
    .when(col("qtd_operadores_associados_ao_cnpj") > 10, "Alta Concentracao")
    .otherwise("NAO_INFORMADO")
)

df = df.withColumn("flag_operador_multicontas",
    when(col("qtd_contas_associadas_ao_operador") > 1, "SIM")
    .otherwise("NAO")
)

# Verificar distribuição
df.groupBy("concetracao_operadores").count().orderBy("concentracao_operadores").show(truncate=False)
df.groupBy("flag_operador_multicontas").count().show()

# ==============================================================================
# 5 - Métricas de observabilidade
# ==============================================================================
# Dataframe de controle - resumo geral do pipeline
metricas = spark.createDataFrame([
    Row(metrica="registros_original", valor=str(total_antes)),
    Row(metrica="duplicatas_removidas", valor=str(duplicatas_removidas)),
    Row(metrica="registros_final", valor=str(total_depois)),
    Row(metrica="total_nulos_antes", valor=str(total_nulos_antes)),
    Row(metrica="total_nulos_corrigidos", valor=str(total_nulos_antes - total_nulos_depois)),
    Row(metrica="total_nulos_restantes", valor=str(total_nulos_depois)),
    Row(metrica="percentual_duplicatas", valor=f"{(duplicatas_removidas / total_antes) * 100:.2f}%"),
    Row(metrica="percentual_nulos_corrigidos",
        valor=f"{((total_nulos_antes - total_nulos_depois) / max(total_nulos_antes, 1)) * 100:.2f}%"),
])

print("=" * 60)
print("MÉTRICAS DE OBSERVABILIDADE - RESUMO DO PIPELINE")
print("=" * 60)
metricas.show(truncate=False)

# Detalhe de nulos por coluna (antes x depois)
linhas_nulos = []
for c in nulos_antes:
    antes = nulos_antes[c]
    depois = nulos_depois.get(c, 0)
    if antes > 0: # Mostrar apenas nulos
        linhas_nulos.append(Row(
            coluna=c,
            nulos_antes=antes,
            nulos_depois=depois,
            corrigidos=antes - depois
        ))

if linhas_nulos:
    print("\nDetalhe de nulos por coluna:")
    df_nulos = spark.createDataFrame(linhas_nulos)
    df_nulos.orderBy("nulos_antes", ascending=False).show(truncate=False)

# Distribuição das colunas transformadas
print("=== faixa_tempo_relacionamento ===")
df.groupBy("faixa_tempo_relacionamento").count().orderBy("faixa_tempo_relacionamento").show(truncate=False)

print("=== classificacao_digital ===")
df.groupBy("classificacao_digital").count().orderBy("classificacao_digital").show(truncate=False)

print("=== score_maturidade_digital ===")
df.groupBy("score_maturidade_digital").count().orderBy("score_maturidade_digital").show()

print("=== risco_operador ===")
df.groupBy("risco_operador").count().orderBy("risco_operador").show()

print("=== concentracao_operadores ===")
df.groupBy("concentracao_operadores").count().orderBy("concentracao_operadores").show(truncate=False)

print("=== flag_operador_multicontas ===")
df.groupBy("flag_operador_multicontas").count().show()

# ==============================================================================
# 6 - Salvamento particionado
# ==============================================================================
# Verificar shema final de salvar
df.printSchema()
print(f"\nTotal de registros a salvar: {df.count()}")

# # --- Opção 1: Salvamento via Spark (requer winutils.exe no Windows) ---
# df.write  \
#     .mode("overwrite")  \
#     .partitionBy("ref_anomes")  \
#     .parquet("output/super_iam_refinado")
# print("Salvo com sucesso via Spark!")

# # --- Opção 2: Salvamento via pyarrow (não requer winutils.exe) ---
# # Necessário: pip install pandas pyarrow
# # import pyarrow as pa
# # import pyarrow.parquet as pq

# pdf = df.toPandas()
# pdf["red_anomes"] = pdf["ref_anomes"].astype(str)

# tabela = pa.Table.from_pandas(pdf)
# pq.write_to_dataset(
#     tabela,
#     root_path="output/super_iam_refinada",
#     partition_cols=["ref_anomes"]
# )

# print("Salvo com sucesso via pyarrow!")
# print("Estrutura: output/super_iam_refinada/ref_anomes=YYYY-MM-DD/")

# # Verificação: Ler o Parquet salvo de volta
# df_salvo = spark.read.parquet("output/super_iam_refinada")
# print(f"Registros salvos: {df_salvo.count()}")
# df_salvo.printSchema()
# df_salvo.show(5, truncate=False)