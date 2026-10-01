# -*- coding: utf-8 -*-
"""
Modelagem de Risco de Credito - Avaliacao de Inadimplencia com Random Forest
Disciplina: Engenharia de Dados e MLOps

Fluxo do script:
  PARTE 1 - Modelo base (dados sinteticos + RandomForest)
  PARTE 2 - Interpretabilidade (importancia das variaveis)
  PARTE 3 - Analise critica (desbalanceamento, matriz de confusao, classification_report)
  PARTE 4 - Desafio extra (modelo balanceado + comparacao e trade-off)

Todas as evidencias (graficos) sao salvas em evidencias/.
Resultados textuais sao salvos em resultados/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend sem interface grafica (salva arquivos sem abrir janela)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Diretorios de saida
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCIAS_DIR = os.path.join(BASE_DIR, "evidencias")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(EVIDENCIAS_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Acumula o resumo final para gravar em resultados/resumo_resultados.txt
LINHAS_RESUMO = []


def log(msg=""):
    """Imprime no terminal e guarda para o arquivo de resumo."""
    print(msg)
    LINHAS_RESUMO.append(str(msg))


# ===========================================================================
# PARTE 1 - MODELO BASE
# ===========================================================================
def gerar_dados(n=2000):
    """
    Gera a base sintetica de clientes EXATAMENTE conforme o enunciado da atividade:
      - RendaMensal       ~ Exponencial(5000) + 1500
      - ScoreCredito      ~ Inteiro uniforme em [300, 950)
      - EmprestimosAtivos ~ Poisson(1.5)

    A variavel alvo Inadimplente e gerada pela regra logistica (sigmoide) do
    enunciado, com coeficientes fixos, usando amostragem estocastica contra uma
    uniforme(0, 1). Nenhum coeficiente, distribuicao ou seed e alterado.
    """
    np.random.seed(42)

    dados = pd.DataFrame(
        {
            "RendaMensal": np.random.exponential(5000, n) + 1500,
            "ScoreCredito": np.random.randint(300, 950, n),
            "EmprestimosAtivos": np.random.poisson(1.5, n),
        }
    )

    prob = 1 / (
        1
        + np.exp(
            -(
                -0.0003 * dados["RendaMensal"]
                - 0.005 * dados["ScoreCredito"]
                + 0.8 * dados["EmprestimosAtivos"]
            )
        )
    )

    dados["Inadimplente"] = (prob > np.random.uniform(0, 1, n)).astype(int)

    return dados


def main():
    log("=" * 70)
    log("MODELAGEM DE RISCO DE CREDITO - RANDOM FOREST")
    log("=" * 70)

    # ---- 1 a 3: gerar dados ------------------------------------------------
    df = gerar_dados(2000)
    df.to_csv(os.path.join(RESULTADOS_DIR, "base_sintetica.csv"), index=False)

    # Conforme o enunciado: X = tudo exceto o alvo; y = alvo
    X = df.drop(columns=["Inadimplente"])
    y = df["Inadimplente"]
    features = list(X.columns)

    log("\n[PARTE 1] Base sintetica gerada")
    log(f"  Total de clientes : {len(df)}")
    log(f"  Variaveis (X)     : {features}")
    log("\n  Primeiras linhas da base:")
    log(df.head().to_string(index=False))

    # ---- 4: split ----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 5: treinar modelo base -------------------------------------------
    modelo_base = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        max_features="sqrt",
        oob_score=True,
        n_jobs=-1,
        random_state=42,
    )
    modelo_base.fit(X_train, y_train)

    # ---- 6: metricas do modelo base ---------------------------------------
    oob = modelo_base.oob_score_
    y_prob_base = modelo_base.predict_proba(X_test)[:, 1]
    y_pred_base = modelo_base.predict(X_test)
    roc_auc_base = roc_auc_score(y_test, y_prob_base)

    log("\n[PARTE 1] Modelo base treinado")
    log(f"  Acuracia OOB        : {oob:.4f}")
    log(f"  ROC-AUC (teste)     : {roc_auc_base:.4f}")

    # =======================================================================
    # PARTE 2 - INTERPRETABILIDADE
    # =======================================================================
    importancias = pd.Series(
        modelo_base.feature_importances_, index=features
    ).sort_values(ascending=False)

    log("\n[PARTE 2] Importancia das variaveis (ordem decrescente):")
    for nome, valor in importancias.items():
        log(f"  {nome:<20}: {valor:.4f}")

    # Grafico de importancia
    plt.figure(figsize=(8, 5))
    importancias.sort_values().plot(kind="barh", color="#4C72B0")
    plt.title("Importancia das Variaveis - Modelo Base")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(os.path.join(EVIDENCIAS_DIR, "importancia_variaveis.png"), dpi=120)
    plt.close()

    # =======================================================================
    # PARTE 3 - ANALISE CRITICA
    # =======================================================================
    prop_inad = y.mean()
    log("\n[PARTE 3] Analise critica do desbalanceamento")
    log(f"  Proporcao de inadimplentes (base completa): {prop_inad:.4f} "
        f"({prop_inad*100:.1f}%)")
    log(f"  Distribuicao de classes: {dict(y.value_counts())}")

    cm_base = confusion_matrix(y_test, y_pred_base)
    report_base = classification_report(y_test, y_pred_base, digits=4)

    log("\n  Matriz de Confusao (modelo base):")
    log(f"    {cm_base.tolist()}")
    log("\n  Classification Report (modelo base):")
    log(report_base)

    # Grafico matriz de confusao - modelo base
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de Confusao - Modelo Base")
    plt.tight_layout()
    plt.savefig(
        os.path.join(EVIDENCIAS_DIR, "matriz_confusao_modelo_base.png"), dpi=120
    )
    plt.close()

    pct_maj = (1 - prop_inad) * 100
    log("\n  >> Por que acuracia elevada pode ser enganosa com desbalanceamento?")
    log(f"     Com ~{pct_maj:.1f}% da base na classe 0, um modelo que SEMPRE preve 'nao")
    log(f"     inadimplente' ja atinge ~{pct_maj:.1f}% de acuracia sem detectar nenhum")
    log("     inadimplente (recall da classe 1 = 0). A acuracia e dominada pela")
    log("     classe majoritaria e mascara a incapacidade de identificar o")
    log("     evento de interesse (classe 1). Por isso usamos precision, recall,")
    log("     f1-score e ROC-AUC, que avaliam o desempenho por classe.")

    # =======================================================================
    # PARTE 4 - DESAFIO EXTRA: MODELO BALANCEADO
    # =======================================================================
    modelo_bal = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        max_features="sqrt",
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    modelo_bal.fit(X_train, y_train)

    oob_bal = modelo_bal.oob_score_
    y_prob_bal = modelo_bal.predict_proba(X_test)[:, 1]
    y_pred_bal = modelo_bal.predict(X_test)
    roc_auc_bal = roc_auc_score(y_test, y_prob_bal)

    cm_bal = confusion_matrix(y_test, y_pred_bal)
    report_bal = classification_report(y_test, y_pred_bal, digits=4)

    log("\n[PARTE 4] Modelo balanceado (class_weight='balanced')")
    log(f"  Acuracia OOB        : {oob_bal:.4f}")
    log(f"  ROC-AUC (teste)     : {roc_auc_bal:.4f}")
    log("\n  Matriz de Confusao (modelo balanceado):")
    log(f"    {cm_bal.tolist()}")
    log("\n  Classification Report (modelo balanceado):")
    log(report_bal)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_bal, display_labels=[0, 1])
    disp.plot(cmap="Oranges", values_format="d")
    plt.title("Matriz de Confusao - Modelo Balanceado")
    plt.tight_layout()
    plt.savefig(
        os.path.join(EVIDENCIAS_DIR, "matriz_confusao_modelo_balanceado.png"), dpi=120
    )
    plt.close()

    # ---- Comparacao focada na classe 1 ------------------------------------
    def metricas_classe1(y_true, y_pred):
        from sklearn.metrics import precision_score, recall_score, f1_score

        return (
            precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        )

    p_base, r_base, f_base = metricas_classe1(y_test, y_pred_base)
    p_bal, r_bal, f_bal = metricas_classe1(y_test, y_pred_bal)

    log("\n[PARTE 4] Comparacao Modelo Base x Modelo Balanceado (classe 1)")
    log(f"  {'Metrica':<20}{'Base':>12}{'Balanceado':>14}")
    log(f"  {'-'*46}")
    log(f"  {'Precision (classe 1)':<20}{p_base:>12.4f}{p_bal:>14.4f}")
    log(f"  {'Recall (classe 1)':<20}{r_base:>12.4f}{r_bal:>14.4f}")
    log(f"  {'F1-score (classe 1)':<20}{f_base:>12.4f}{f_bal:>14.4f}")
    log(f"  {'ROC-AUC':<20}{roc_auc_base:>12.4f}{roc_auc_bal:>14.4f}")

    # Grafico comparacao de recall da classe 1
    plt.figure(figsize=(6, 5))
    plt.bar(["Base", "Balanceado"], [r_base, r_bal], color=["#4C72B0", "#DD8452"])
    plt.ylim(0, 1)
    plt.ylabel("Recall (classe 1)")
    plt.title("Comparacao de Recall da Classe 1 (Inadimplentes)")
    for i, v in enumerate([r_base, r_bal]):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(EVIDENCIAS_DIR, "comparacao_recall.png"), dpi=120)
    plt.close()

    # ---- Trade-off ---------------------------------------------------------
    log("\n[PARTE 4] Analise do trade-off (nao basta dizer qual e 'melhor'):")
    log("  O class_weight='balanced' penaliza mais os erros na classe minoritaria")
    log("  (inadimplentes). O efeito tipico e AUMENTAR o recall da classe 1")
    log("  (detectar mais inadimplentes), ao custo de REDUZIR a precision da")
    log("  classe 1 (mais falsos positivos: bons pagadores sinalizados como risco).")
    log("  A escolha depende do custo de negocio: se deixar passar um inadimplente")
    log("  (falso negativo) e mais caro que recusar credito a um bom pagador")
    log("  (falso positivo), o modelo balanceado tende a ser preferivel; caso")
    log("  contrario, o ganho em recall pode nao compensar a perda de precisao.")

    # ---- Resumo final + gravacao ------------------------------------------
    log("\n" + "=" * 70)
    log("RESUMO FINAL")
    log("=" * 70)
    log(f"  Inadimplentes na base      : {prop_inad*100:.1f}%")
    log(f"  OOB base / balanceado      : {oob:.4f} / {oob_bal:.4f}")
    log(f"  ROC-AUC base / balanceado  : {roc_auc_base:.4f} / {roc_auc_bal:.4f}")
    log(f"  Recall cl.1 base/balanceado: {r_base:.4f} / {r_bal:.4f}")
    log(f"  Precision cl.1 base/balanc.: {p_base:.4f} / {p_bal:.4f}")
    log(f"  F1 cl.1 base / balanceado  : {f_base:.4f} / {f_bal:.4f}")
    log("\n  Evidencias salvas em: evidencias/")
    log("    - importancia_variaveis.png")
    log("    - matriz_confusao_modelo_base.png")
    log("    - matriz_confusao_modelo_balanceado.png")
    log("    - comparacao_recall.png")
    log("  Resultados salvos em: resultados/")
    log("    - base_sintetica.csv")
    log("    - resumo_resultados.txt")

    # Grava relatorios textuais
    with open(
        os.path.join(RESULTADOS_DIR, "classification_report_base.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_base)
    with open(
        os.path.join(RESULTADOS_DIR, "classification_report_balanceado.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_bal)
    with open(
        os.path.join(RESULTADOS_DIR, "resumo_resultados.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(LINHAS_RESUMO))


if __name__ == "__main__":
    main()
