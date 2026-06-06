def visualize_text_int(
    rank_idx: int,
    row_idx: int,
    col_idx: int,
    dataset,
    llm,
    crosscoder,
    number_of_examples: int = 2,
):
    """
    Show the top-N examples of the (row_idx, col_idx) interaction.
    On each example we render three side‑by‑side highlights:
      1) the interaction strength at (row_idx, col_idx)
      2) the row feature activation
      3) the col feature activation
    """
    texts = [dataset[i]["text"] for i in range(number_of_examples)]
    rows_html = []

    for txt in texts:
        # --- 1) get per‑token interaction scores at (row_idx, col_idx) ---
        ints_all = feature_interactions_mlp(
            txt, llm, crosscoder, block=1        # <-- named parameter so block=1 actually takes effect
        )                                        # shape [seq_len, num_feats, num_feats]
        ints_tok = ints_all[:, row_idx, col_idx] \
                      .detach().cpu().numpy()     # shape [seq_len]

        # --- 2) get the two individual feature activations ---
        acts = get_activations(txt, llm, crosscoder)[0]  # shape [seq_len, num_feats]
        row_tok = acts[:, row_idx].detach().cpu().numpy()
        col_tok = acts[:, col_idx].detach().cpu().numpy()

        # --- 3) encode to tokens (must match len(ints_tok)) ---
        tokens = llm.tokenizer.encode(txt)[: len(ints_tok)]

        # --- 4) render three side‑by‑side snippets via the low‑level highlighter ---
        html_int = display_text_with_highlighting(
            tokens, llm.tokenizer, ints_tok, transparent_test=lambda v: v == 0
        )
        html_row = display_text_with_highlighting(
            tokens, llm.tokenizer, row_tok, transparent_test=lambda v: v == 0
        )
        html_col = display_text_with_highlighting(
            tokens, llm.tokenizer, col_tok, transparent_test=lambda v: v == 0
        )

        rows_html.append(
            f"<div style='display:flex; gap:10px; margin-bottom:20px;'>"
            f"{html_int.data}{html_row.data}{html_col.data}"
            f"</div>"
        )

    section_html = f"""
    <section id="feature_int_{rank_idx}" style="display:none">
      <h2>Interaction {row_idx} → {col_idx}</h2>
      {''.join(rows_html)}
    </section>
    """
    return section_html 