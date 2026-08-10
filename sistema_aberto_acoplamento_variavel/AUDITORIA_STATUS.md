# Acompanhamento das correções — Open TDJC

Última atualização: **2026-08-06**

Arquivo principal: `paper_draft.tex`

Este arquivo é o checklist mestre das correções. Ele separa o estado do **cálculo**, das **figuras** e do **texto do artigo**, pois uma conta corrigida não significa que o artigo já tenha sido atualizado.

## Legenda de status

| Símbolo | Status | Significado |
|---|---|---|
| ✅ | Concluído | Verificado e já resolvido no nível indicado. |
| 🟡 | Em incorporação | Cálculo ou decisão já existe, mas ainda falta atualizar figuras e/ou artigo. |
| 🔴 | Pendente prioritário | Pode alterar resultados ou interpretação científica. |
| ⚪ | Decisão necessária | É necessário escolher uma convenção ou abordagem antes de continuar. |
| 🔵 | Verificação necessária | Há uma hipótese plausível, mas é preciso confirmar nos arquivos/resultados finais. |
| ➖ | Sem alteração | Item verificado; não requer mudança neste momento. |

## Situação consolidada atual

| Área | Situação confirmada |
|---|---|
| Taxas | O artigo e as contas novas usam `kappa = 1e-1` e `gamma_phi = 1e-2`. As contas novas ainda não foram incorporadas ao artigo. |
| Coerência | A implementação/cálculo já foi corrigida, mas as Figs. 7–9 e a discussão atual do artigo ainda correspondem aos resultados antigos. |
| Magic | A fórmula de `M2` foi realmente usada, mas ainda precisa ser resolvida a interpretação de magic para o qubit reduzido misto. |
| C5/C6 | Os erros foram identificados, mas ainda não foram corrigidos no `paper_draft.tex`. |
| Escopo | Neste momento, o acompanhamento é do artigo e dos resultados; não editar códigos sem uma decisão explícita. |

## Ordem de trabalho recomendada

| Ordem | Ação | Critério para concluir | Status |
|---:|---|---|---|
| 1 | Comparar a coerência corrigida com as Figs. 7–9 atuais | Confirmar quais conclusões qualitativas permanecem e quais mudam | 🟡 Em incorporação |
| 2 | Substituir no artigo as Figs. 7–9 e revisar a Sec. III B | Figuras novas inseridas; parágrafos consistentes com elas | 🔴 Pendente prioritário |
| 3 | Resolver a medida/interpretação de magic para estados mistos | Observable escolhido e afirmações cientificamente sustentadas | ⚪ Decisão necessária |
| 4 | Resolver a divergência do estado inicial de magic | Alterar a Eq. para o estado efetivamente simulado ou rerodar usando o estado atualmente escrito | ⚪ Decisão necessária |
| 5 | Incorporar os demais resultados novos que forem aprovados | Cada figura rastreada ao diretório e aos parâmetros corretos | 🔵 Verificação necessária |
| 6 | Corrigir C5, C6, eixos e legendas copiadas | Convenção de `omega` uniforme nas Figs. 3, 9, 12 e 15 | 🔴 Pendente prioritário |
| 7 | Corrigir métodos, notação, resumo e conclusões | Texto final coerente com as figuras aprovadas | 🔴 Pendente prioritário |
| 8 | Fazer revisão final de reprodutibilidade e idioma | Sem comentários internos; métodos e disponibilidade documentados | 🔴 Pendente prioritário |

## Tabela mestre de mudanças

| ID | Prioridade | Local/tema | Problema ou decisão | Mudança necessária | Estado do cálculo/dado | Estado no artigo | Status geral |
|---|---|---|---|---|---|---|---|
| T01 | P0 | Taxas dissipativas | O PDF de especificação antigo mencionava `1e-2/1e-3`, mas o artigo e as contas novas usam `kappa=1e-1`, `gamma_phi=1e-2` | Manter `1e-1/1e-2`, salvo nova decisão explícita; não chamar esses valores de erro do artigo | Confirmado nas contas novas | O texto atual já declara `1e-1/1e-2` | ➖ Sem alteração |
| T02 | P1 | Convenção de dephasing | Com o colapso `sqrt(gamma_phi) sigma_z`, a coerência isolada decai como `exp(-2 gamma_phi t)` | Explicar a convenção nos métodos; com `gamma_phi=1e-2`, `tau_phi=50` | Confirmado pelo código | Não explicado claramente | 🟡 Em incorporação |
| T03 | P1 | Tempos dissipativos | Para `kappa=1e-1`, `tau_n=10` e `tau_a=20` | Incluir somente se os tempos forem usados na interpretação física | Confirmado | Ausente | 🟡 Em incorporação |
| C01 | P0 | Coerência — implementação | A rotina antiga calculava a entropia sobre `rho^2` | Usar `C_rel(rho)=S(rho_diag)-S(rho)` sobre `rho` | ✅ Corrigido nas contas novas | Resultados antigos ainda presentes | 🟡 Em incorporação |
| C02 | P0 | Figs. 7–9 | As figuras atuais ainda são da coerência antiga | Comparar e substituir pelos gráficos corrigidos | Novos cálculos disponíveis | Não atualizado | 🔴 Pendente prioritário |
| C03 | P0 | Sec. III B | A discussão pode não corresponder às curvas corrigidas | Revisar cada afirmação após comparar as figuras antigas e novas | A verificar nas curvas novas | Texto antigo | 🔴 Pendente prioritário |
| C04 | P1 | Base do log da coerência | Texto e implementação precisam usar a mesma base | Preferencialmente usar `log2`, ou declarar outra base de forma consistente | 🔵 Confirmar na versão corrigida | A conferir | 🔵 Verificação necessária |
| C05 | P1 | Normalização da coerência | Curvas normalizadas precisam ter definição explícita | Definir, por exemplo, `C_tilde(t)=C(t)/C(0)` ou a normalização realmente usada | 🔵 Confirmar nos gráficos novos | Não definido com precisão | 🔵 Verificação necessária |
| C06 | P1 | Caso médio das Figs. 7 e 8 | O mesmo ponto exato `zeta=8, T=25` deveria produzir a mesma dinâmica | Comparar valores exatos, antes da normalização; não usar apenas índice central da grade | Contas específicas existem; conferir | Comentário interno ainda sinaliza dúvida | 🔵 Verificação necessária |
| M01 | P0 | Magic — fórmula usada | Era necessário confirmar se o código realmente usou a fórmula de `M2` | Registrar que foi usado `M2=ln(S2/S4)`, com `M2(0)=ln(3/2)` | ✅ Confirmado | Fórmula presente | ✅ Concluído |
| M02 | P0 | Magic de estado misto | `M2>0` sozinho não certifica magic para o qubit reduzido misto | Escolher um critério válido: octaedro estabilizador, witness misto ou outro monótono | Diagnóstico preliminar mostra estados finais não mágicos apesar de `M2>0` | Afirmações atuais tratam `M2` como magic preservada | ⚪ Decisão necessária |
| M03 | P0 | Estado inicial de magic | O artigo usa `cos(theta/2)|g> + exp(i pi/4) sin(theta/2)|e>`, enquanto as contas antigas e novas usam `cos(theta/2)|e> + exp(i pi/4) sin(theta/2)|g>` | Como ambos são maximamente mágicos inicialmente, a opção mínima é alterar a Eq. para o estado efetivamente simulado; se o estado escrito for intencional, é necessário rerodar | ✅ Divergência confirmada nas contas antigas e novas | Eq. atual não corresponde ao estado simulado | ⚪ Decisão necessária |
| M04 | P0 | Figs. 10–12 | Podem precisar ser regeneradas após a decisão sobre estados mistos e sobre o estado inicial | Regenerar depois de escolher o observable; quanto ao estado, não há rerun se a Eq. for alterada para corresponder às contas | Aguardando decisão | Figuras antigas presentes | ⚪ Decisão necessária |
| M05 | P1 | Legendas das Figs. 10–12 | Algumas legendas usam `N(rho)`, símbolo da negatividade de Wigner | Trocar pelo símbolo exato do observable de magic finalmente escolhido | Não se aplica | Pendente | 🔴 Pendente prioritário |
| M06 | P2 | Texto de magic | Há “Réniy” e “normalized normalized” | Corrigir para “Rényi” e remover duplicação | Não se aplica | Pendente | 🔴 Pendente prioritário |
| W01 | P1 | Eq. da negatividade de Wigner | O código usa `1/2 (integral |W| - 1)`, mas falta `1/2` no artigo | Alinhar a equação escrita com a implementação | Código verificado | Pendente | 🔴 Pendente prioritário |
| W02 | P1 | Estado cat | O artigo usa `1/sqrt(2)`, enquanto o código normaliza exatamente | Escrever a normalização exata ou declarar a aproximação para `alpha=sqrt(5)` | Código correto | Pendente | 🟡 Em incorporação |
| W03 | P2 | Corte de Fock | `Nb=45` não está documentado | Informar `Nb=45` e resumir o teste de convergência | Corte considerado suficiente para `alpha=sqrt(5)` | Ausente | 🔴 Pendente prioritário |
| W04 | P2 | Grade de Wigner | Grade `[-7.5,7.5] x [-7.5,7.5]`, `200 x 200`, não documentada | Incluir nos métodos e confirmar convergência em estados evoluídos | Parcialmente verificado | Ausente | 🔵 Verificação necessária |
| E01 | P1 | Estado inicial de emaranhamento | O sinal de `alpha` aparece trocado entre texto e código | Alinhar a convenção para reprodutibilidade | Métrica provavelmente invariante a essa troca | Pendente | 🟡 Em incorporação |
| E02 | P1 | Valor inicial de `E_N` | Para `alpha=sqrt(5)`, o valor é praticamente, mas não exatamente, 1 | Escrever “approximately maximally entangled” ou `E_N(0) approximately 1` | ✅ Verificado | Pendente | 🔴 Pendente prioritário |
| E03 | P1 | Interpretação de `E_N=0` | PPT não implica separabilidade geral em dimensão `2 x 45` | Manter a formulação “no entanglement detected by logarithmic negativity” | Conceitualmente verificado | Conferir redação final | 🟡 Em incorporação |
| A01 | P1 | Amplitude `alpha` | O artigo antigo dizia `alpha=1`; o atual já usa `sqrt(5)` | Manter `alpha=sqrt(5)` e `mean n=5` | ✅ Coerente com os resultados | Valor corrigido | ✅ Concluído |
| A02 | P1 | Motivação de `alpha` | O comentário aproxima `|n=1>` por `|alpha=sqrt(5)>`, o que é incorreto | Remover essa motivação; `P(n=1)=5 exp(-5) approximately 0.0337` | Verificado | Comentário interno ainda presente | 🔴 Pendente prioritário |
| O01 | P1 | C6 — Sec. III B/Fig. 9 | O texto diz `6 < omega < 10`, faixa copiada de `zeta` | Escrever que a varredura completa é `0 <= omega <= pi/5` | Confirmado nos parâmetros | Pendente | 🔴 Pendente prioritário |
| O02 | P1 | Valores destacados de `omega` | `pi/20` é o menor valor destacado, não o mínimo da varredura | Dizer que o painel mostra `pi/20`, `pi/10`, `pi/5` | Confirmado | Pendente | 🔴 Pendente prioritário |
| O03 | P1 | C5 — Fig. 15 | A legenda diz `omega_av=8` | Remover `8`; identificar `pi/10` como ponto médio da faixa completa `[0,pi/5]` | As contas específicas usam `pi/10` | Pendente | 🔴 Pendente prioritário |
| O04 | P1 | Média alternativa | A média de `pi/20` e `pi/5` seria `pi/8`, mas esse não foi o valor específico escolhido | Não escrever `(omega_min+omega_max)/2` se os extremos forem os destacados; usar “selected intermediate value” | Confirmado | Pendente | 🔴 Pendente prioritário |
| O05 | P1 | Figuras cosseno | Eixos mostram `omega/5` ou `omega/3`, mas os ticks correspondem à multiplicação | Preferir eixo físico `omega`; alternativamente usar `5 omega` ou `3 omega` corretamente | Confirmado visualmente | Pendente | 🔴 Pendente prioritário |
| O06 | P2 | Valores exatos versus grade | Grades pares não contêm exatamente alguns pontos médios | Usar as simulações específicas exatas nas curvas destacadas e registrar a origem | Casos específicos existem | Não explicado | 🔵 Verificação necessária |
| G01 | P1 | Definição da gaussiana | `T` é centro/instante de pico; `zeta` controla a largura | Corrigir qualquer frase que atribua a largura a `T` | Código verificado | Pendente | 🔴 Pendente prioritário |
| G02 | P1 | Afirmação `g(0)=0` | Para a gaussiana, `g(0)` é pequeno em vários casos, mas não identicamente zero | Escrever “weak/negligible initial coupling” e quantificar quando necessário | Valores verificados | Pendente | 🔴 Pendente prioritário |
| D01 | P1 | Canais dissipativos | O texto fala em três tipos, mas apenas damping da cavidade e dephasing estão ativos; `gamma=0` | Declarar dois canais ativos e `gamma=0`; corrigir índices `L_2/L_3` | Código verificado | Pendente | 🔴 Pendente prioritário |
| D02 | P1 | Símbolos das taxas | Artigo usa `gamma` para dephasing; código usa `gamma_phi`, enquanto `gamma` é emissão espontânea | Criar mapeamento explícito e evitar usar o mesmo símbolo para canais diferentes | Confirmado | Pendente | 🔴 Pendente prioritário |
| R01 | P1 | Número de recursos | Resumo e abertura da Sec. III falam em três recursos, mas o artigo contém quatro | Incluir magic e escrever “four resources” | Não se aplica | Pendente | 🔴 Pendente prioritário |
| R02 | P1 | Resumo | A afirmação geral de que a modulação trigonométrica preserva melhor não vale igualmente para todos os recursos | Condicionar a conclusão por recurso após aprovar as figuras novas | Depende dos resultados finais | Pendente | ⚪ Decisão após figuras |
| R03 | P1 | Conclusões | A seção de conclusões está vazia | Escrever somente após incorporar e validar todas as figuras novas | Aguardando resultados finais | Pendente | 🔴 Pendente prioritário |
| R04 | P1 | Limiar | O artigo anuncia limiar `1e-2`, mas não apresenta tempos de cruzamento; scripts antigos mencionam `1e-1` | Remover a promessa ou definir e reportar uma medida de persistência robusta | Não consolidado | Pendente | ⚪ Decisão necessária |
| R05 | P2 | Normalização das figuras | Colorbars/eixos frequentemente mostram quantidade bruta embora os dados estejam normalizados | Definir cada quantidade normalizada com til e denominador explícito | 🔵 Conferir nas figuras novas | Pendente | 🔵 Verificação necessária |
| N01 | P2 | Métodos numéricos | Faltam solver, versões, tolerâncias e grade temporal | Documentar QuTiP `mesolve`, versões, `atol/rtol`, pontos de saída e `tmax` | Dados parcialmente disponíveis | Ausente | 🔴 Pendente prioritário |
| N02 | P2 | Convergência temporal | Passos de saída ficam mais espaçados após `t approximately 1` | Testar uma amostra com grade mais densa e tolerâncias mais estritas | Não certificado para a versão final | Ausente | 🔵 Verificação necessária |
| N03 | P2 | Proveniência das figuras | Algumas figuras antigas têm rastreabilidade desigual, especialmente a coerência cosseno | Registrar para cada figura: diretório, parâmetros, data/versão e arquivo de saída | Parcial | Ausente | 🔵 Verificação necessária |
| N04 | P2 | Disponibilidade | Falta declaração clara de código e dados | Adicionar seção de data/code availability | Repositório existe | Ausente | 🔴 Pendente prioritário |
| X01 | P1 | Comentários internos | Permanecem comentários `Cesar/Thiago` no corpo e nas legendas | Resolver o conteúdo e remover todos antes da submissão | Não se aplica | Pendente | 🔴 Pendente prioritário |
| X02 | P2 | Revisão de idioma | Há typos, concordância e frases copiadas | Fazer revisão final somente após estabilizar conteúdo e figuras | Não se aplica | Pendente | 🔴 Pendente prioritário |

## Registro das atualizações

| Data | Atualização | Consequência no plano |
|---|---|---|
| 2026-08-06 | Confirmado que as contas novas usam `kappa=1e-1` e `gamma_phi=1e-2` e ainda não estão no artigo | As taxas deixam de ser a primeira correção; mantê-las por enquanto |
| 2026-08-06 | Confirmado que a coerência já foi corrigida nas contas novas | A prioridade imediata passa a ser comparar e incorporar as novas Figs. 7–9 |

## Próxima ação

**Comparar, uma a uma, as figuras corrigidas de coerência com as Figs. 7–9 atuais e registrar nesta tabela quais interpretações da Sec. III B permanecem válidas.**
