"""PROJECT & RESULTS: an editorial report with a contextual section index.

Replaces the former Home, Performance, Model Comparison and About pages.
Every number is read from outputs/metrics/*.json, models/*/model_info.json
or the FNC-1 CSVs; nothing is typed in by hand.
"""
from __future__ import annotations

import math

import streamlit as st

import config
from ui import theme as T
from ui.common import dataset_stats, engine_status, fmt_pct, historical_metrics, model_info, ordered_keys, ours_metrics

SECTIONS = [
    ("overview", "Overview"),
    ("performance", "Model performance"),
    ("comparison", "Model comparison"),
    ("methodology", "Methodology"),
    ("dataset", "Dataset"),
    ("limitations", "Limitations & credits"),
]
SECTION_KEYS = [k for k, _ in SECTIONS]


def _set_section(key: str) -> None:
    st.session_state["section"] = key


def render() -> None:
    ss = st.session_state
    if ss.get("section") not in SECTION_KEYS:
        ss["section"] = "overview"
    current = ss["section"]

    with st.container(key="results_shell"):
        idx, body = st.columns([1, 4.2], gap="large")
        with idx:
            with st.container(key="section_index"):
                T.md('<div class="sd-idx-h">Project &amp; results · Index</div>')
                for i, (key, title) in enumerate(SECTIONS, 1):
                    state = "on" if key == current else "off"
                    st.button(f"{i:02d}   {title}", key=f"sec_{key}_{state}", on_click=_set_section, args=(key,),
                              use_container_width=True)
                T.md('<div class="sd-idx-foot">Figures are read from <code>outputs/metrics/</code> '
                     "and recomputed by <code>evaluation.py</code>; none are typed in by hand.</div>")
        with body:
            with st.container(key="section_body"):
                {"overview": _overview, "performance": _performance, "comparison": _comparison,
                 "methodology": _methodology, "dataset": _dataset, "limitations": _limitations}[current]()
        T.md('<div class="sd-foot"><span>The Stance Desk · FNC-1 stance detection</span>'
             "<span>Stance detection is not fact-checking.</span></div>")


def _num(i: str) -> int:
    return SECTION_KEYS.index(i) + 1


# ---------------------------------------------------------------------------
# 01 Overview
# ---------------------------------------------------------------------------
def _overview() -> None:
    T.md(T.section_head(
        _num("overview"), "Overview", "Does the article back the headline?",
        "Misinformation spreads faster than it can be checked by hand. The Fake News Challenge (FNC-1) split the "
        "problem into a first, automatable step: <em>stance detection</em>. Given a claim and an article, does "
        "the article support it, refute it, discuss it, or ignore it?"))
    T.md(
        '<div class="sd-two"><div class="sd-prose">'
        "<p><strong>What stance detection is.</strong> Fake-news investigations often start by asking <em>what do "
        "other sources say about this claim?</em> Stance detection automates that first step: it reads an "
        "article and decides whether it supports, contradicts or merely reports the headline's claim, or "
        "whether it is about something else entirely.</p>"
        "<p><strong>Why the pair matters.</strong> The same article can <em>agree</em> with one headline and be "
        "<em>unrelated</em> to another. The label is a property of the <strong>pair</strong>, so a model must "
        "compare the two texts rather than look at either one alone.</p>"
        "</div><div class=\"sd-prose\">"
        "<p><strong>What this project solves.</strong> It is a working, re-implemented and audited version of "
        "an FNC-1 stance detector: a BERT model fine-tuned here, a Logistic Regression engine as CPU "
        "fallback, honest re-scoring of the original repository's results, and this interface for trying "
        "the models on any headline and article.</p>"
        '<div class="sd-callout"><p><b>Stance detection is not fact-checking.</b> The desk judges how an '
        "article relates to a headline. An article that agrees with a claim can still be wrong, and one that "
        "disagrees can be wrong too. It helps a fact-checker find the articles that matter; it does not "
        "decide what is true.</p></div>"
        "</div></div>"
    )
    T.md(T.h3("The four stances", "FNC-1 label set"))
    cells = "".join(
        f'<div class="sd-c-{l}"><span class="sd-mini">{l.upper()}</span><p>{T.esc(config.LABEL_EXPLANATIONS[l])}</p></div>'
        for l in config.DISPLAY_ORDER)
    T.md(f'<div class="sd-classes">{cells}</div>')

    ours = ours_metrics()
    if "bert_fnc1" in ours:
        m = ours["bert_fnc1"]["metrics"]
        T.md(T.h3("Headline figures", "fine-tuned BERT · official FNC-1 competition test set"))
        T.md(_ledger([
            (fmt_pct(m["accuracy"]), "Accuracy", f"always-unrelated: {fmt_pct(m['majority_baseline_accuracy'])}"),
            (f"{m['macro']['f1']:.3f}", "Macro-F1", "unweighted mean of 4 classes"),
            (fmt_pct(m["fnc"]["relative"]), "FNC-1 score", f"{m['fnc']['score']:,.2f} of {m['fnc']['max']:,.2f}"),
            (f"{m['per_class']['disagree']['f1']:.3f}", "Disagree F1", "rarest, hardest class"),
            (f"{m['n']:,}", "Test pairs", "scored once, after selection"),
        ]))

    T.md(T.h3("Prediction engines", "status of the models on this machine"))
    engines = engine_status()
    rows = []
    for kind, key, title, desc in (
        ("bert", "bert_fnc1", "BERT (bert-base-uncased, fine-tuned here)", "Default engine. Uses the GPU when available."),
        ("lr", "lr_relational", "Logistic Regression (TF-IDF + relation features)", "CPU fallback engine."),
    ):
        if kind in engines:
            info = model_info(kind)
            status = '<span class="sd-badge sd-badge--ours">Ready</span>'
            extra = []
            if kind == "bert" and info:
                extra.append(f"epochs {info.get('epochs_completed')} · max {info.get('max_length')} tokens · "
                             f"{info.get('optimizer_steps')} optimizer steps")
            if key in ours:
                mm = ours[key]["metrics"]
                extra.append(f"test accuracy {fmt_pct(mm['accuracy'])} · macro-F1 {mm['macro']['f1']:.3f} · "
                             f"FNC {fmt_pct(mm['fnc']['relative'])}")
            else:
                extra.append("test-set evaluation not yet available")
        else:
            status = '<span class="sd-badge sd-badge--hist">Not available</span>'
            extra = ["model files not found"]
        rows.append(f"<tr><td>{T.esc(title)}<span class=\"sd-note\">{T.esc(desc)}</span></td>"
                    f"<td style=\"text-align:left\">{status}</td>"
                    f"<td style=\"text-align:left;white-space:normal\">{'<br>'.join(T.esc(e) for e in extra)}</td></tr>")
    T.md('<div class="sd-table-wrap"><table class="sd-table"><thead><tr><th>Engine</th>'
         '<th style="text-align:left">Status</th><th style="text-align:left">Details</th></tr></thead>'
         f"<tbody>{''.join(rows)}</tbody></table></div>")


def _ledger(items) -> str:
    cells = "".join(f'<div><div class="sd-fig">{v}</div><div class="sd-cap">{T.esc(c)}<span>{T.esc(s)}</span></div></div>'
                    for v, c, s in items)
    return f'<div class="sd-ledger" style="--n:{len(items)}">{cells}</div>'


# ---------------------------------------------------------------------------
# 02 Performance
# ---------------------------------------------------------------------------
def _performance() -> None:
    T.md(T.section_head(
        _num("performance"), "Model performance", "How the models score",
        "All numbers are computed from saved predictions by <code class=\"sd-code\">evaluation.py</code>. "
        "None are typed in by hand."))
    ours, hist = ours_metrics(), historical_metrics()
    with st.container(key="perf_controls"):
        c1, c2 = st.columns([1.25, 1])
        with c1:
            src = st.segmented_control(
                "Source", ["ours", "hist"], key="perf_source", default="ours", required=True,
                format_func={"ours": "Trained here", "hist": "Historical · original repo"}.get,
                help="Trained here = models trained and evaluated in this project. Historical = metrics recomputed "
                     "from the original repository's committed predictions (not reproduced).")
        src = src or "ours"
        data = ours if src == "ours" else hist
        with c2:
            if data:
                keys = ordered_keys(data)
                skey = f"perf_model_{src}"
                if st.session_state.get(skey) not in keys:
                    st.session_state[skey] = keys[0]
                key = st.selectbox("Model", keys, key=skey,
                                   format_func=lambda k: data[k]["meta"].get("display_name", k))
    if not data:
        T.md('<div class="sd-prov sd-prov--hist">'
             + ("No results from our implementation yet. Run the training and evaluation scripts (README)."
                if src == "ours" else "Historical metrics not found. Run <code>python evaluate_historical.py</code>.")
             + "</div>")
        return

    if src == "ours":
        T.md(f'<div class="sd-prov">{T.badge("ours")}<span>Trained and evaluated in this project on the official '
             "FNC-1 competition test set (used once, after model selection on a validation split grouped by "
             "article body).</span></div>")
    else:
        meta = data[key]["meta"]
        T.md(f'<div class="sd-prov sd-prov--hist">{T.badge("hist")}<span><b>Historical, self-reported and not '
             "reproduced.</b> These metrics are recomputed from the prediction files committed in the original "
             "repository. Its BERT checkpoint is not public, so the results cannot be regenerated. File names "
             "for the two Logistic Regression baselines were swapped in the original repository; they are "
             "labelled here by the features actually used.<br>"
             f'<span class="sd-code">output/{T.esc(meta.get("file"))}</span> · SHA-256 '
             f'<span class="sd-code">{T.esc(meta.get("sha256", "")[:16])}…</span></span></div>')
    _metric_block(data[key])


def _metric_block(entry: dict) -> None:
    m, meta = entry["metrics"], entry["meta"]
    T.md(_ledger([
        (fmt_pct(m["accuracy"]), "Accuracy", f"n = {m['n']:,}"),
        (f"{m['macro']['precision']:.3f}", "Precision (macro)", "mean over 4 classes"),
        (f"{m['macro']['recall']:.3f}", "Recall (macro)", "mean over 4 classes"),
        (f"{m['macro']['f1']:.3f}", "Macro-F1", "headline metric"),
        (fmt_pct(m["fnc"]["relative"]), "FNC-1 score", f"raw {m['fnc']['score']:,.2f} / {m['fnc']['max']:,.2f}"),
    ]))
    note = f" · note: {T.esc(meta['note'])}" if meta.get("note") else ""
    T.md(f'<div class="sd-fine">Evaluated on: {T.esc(meta.get("evaluated_on", "n/a"))} · '
         f"'always unrelated' accuracy = {fmt_pct(m['majority_baseline_accuracy'])}, "
         f"FNC = {fmt_pct(m['fnc']['null_relative'])}{note}. Relative FNC-1 score = weighted score / maximum "
         "possible score.</div>")

    T.md('<div class="sd-two sd-two--cm" style="margin-top:.6rem">'
         f"<div>{T.h3('Confusion matrix', 'row-normalised; counts shown')}{_confusion(m)}</div>"
         f"<div>{T.h3('Per-class performance', 'test set')}{_per_class(m)}"
         f"{T.h3('Aggregates')}"
         '<dl class="sd-dl">'
         f"<dt>Weighted P / R / F1</dt><dd>{m['weighted']['precision']:.3f} / {m['weighted']['recall']:.3f} / "
         f"{m['weighted']['f1']:.3f}</dd>"
         f"<dt>Related vs unrelated</dt><dd>{fmt_pct(m['related_vs_unrelated_accuracy'])} accuracy</dd>"
         f"<dt>Raw FNC-1 score</dt><dd>{m['fnc']['score']:,.2f} out of a maximum {m['fnc']['max']:,.2f}</dd>"
         f"<dt>Always unrelated</dt><dd>{m['fnc']['null']:,.2f} FNC ({fmt_pct(m['fnc']['null_relative'])})</dd>"
         "</dl></div></div>")

    vm = meta.get("validation_metrics")
    if vm:
        T.md(T.h3("Validation split", "used for model selection, not the test set")
             + f'<div class="sd-fine" style="margin-top:0">Accuracy {fmt_pct(vm["accuracy"])} · macro-F1 '
             f"{vm['macro']['f1']:.3f} · FNC {fmt_pct(vm['fnc']['relative'])} · n = {vm['n']:,}</div>")


def _confusion(m: dict) -> str:
    cm = m["confusion_matrix"]
    labels, z = cm["labels"], cm["rows_gold_cols_pred"]
    out = ['<div class="sd-cm" role="table" aria-label="Confusion matrix, rows gold, columns predicted">',
           '<div class="sd-cm-h">gold → pred</div>']
    out += [f'<div class="sd-cm-h" style="color:{T.CLASS_COLORS.get(l, "")}">{l}</div>' for l in labels]
    for i, row in enumerate(z):
        total = max(1, sum(row))
        out.append(f'<div class="sd-cm-r" style="color:{T.CLASS_COLORS.get(labels[i], "")}">{labels[i]}</div>')
        for j, v in enumerate(row):
            share = v / total
            diag = i == j
            base = "77,120,168" if diag else "210,58,50"
            alpha = 0.08 + 0.72 * share
            out.append(f'<div class="sd-cell{" sd-cell--diag" if diag else ""}" '
                       f'style="background:rgba({base},{alpha:.3f})" title="gold {labels[i]} → predicted {labels[j]}">'
                       f"<b>{v:,}</b><span>{share:.0%}</span></div>")
    out.append("</div>")
    out.append('<div class="sd-axis"><span>Blue diagonal = correct</span><span>Red = confusions</span></div>')
    return "".join(out)


def _per_class(m: dict) -> str:
    rows = []
    for lab in config.DISPLAY_ORDER:
        v = m["per_class"].get(lab)
        if not v:
            continue
        c = T.CLASS_COLORS[lab]
        rows.append(
            f'<tr style="--c:{c}"><td><span class="sd-dot"></span>{lab}</td>'
            f"<td>{v['precision']:.3f}</td><td>{v['recall']:.3f}</td>"
            f"<td>{v['f1']:.3f}<span class=\"sd-inbar\"><i style=\"width:{v['f1'] * 100:.1f}%\"></i></span></td>"
            f"<td>{v['support']:,}</td><td>{v['predicted']:,}</td></tr>")
    return ('<div class="sd-table-wrap"><table class="sd-table"><thead><tr><th>Class</th><th>Precision</th>'
            "<th>Recall</th><th>F1</th><th>Support</th><th>Predicted</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# ---------------------------------------------------------------------------
# 03 Comparison
# ---------------------------------------------------------------------------
def _comparison_rows() -> list[dict]:
    rows = []
    for src, data in (("ours", ours_metrics()), ("hist", historical_metrics())):
        for k in ordered_keys(data):
            v = data[k]
            m = v["metrics"]
            rows.append({"src": src, "name": v["meta"].get("display_name", k),
                         "sub": v["meta"].get("features") or v["meta"].get("trained_on", ""),
                         "acc": m["accuracy"], "f1": m["macro"]["f1"], "p": m["macro"]["precision"],
                         "r": m["macro"]["recall"], "dis": m["per_class"]["disagree"]["f1"],
                         "fnc": m["fnc"]["relative"], "raw": m["fnc"]["score"]})
    for src in ("ours", "hist"):
        part = [r for r in rows if r["src"] == src]
        part.sort(key=lambda r: -r["f1"])
        rows = [r for r in rows if r["src"] != src] + part
    return rows


def _comparison() -> None:
    T.md(T.section_head(
        _num("comparison"), "Model comparison", "Trained here, measured honestly",
        "Every row is scored on the same official FNC-1 competition test set with the same evaluation code. "
        "Rows trained in this project and historical rows from the original repository are kept apart."))
    rows = _comparison_rows()
    if not rows:
        T.md('<div class="sd-prov sd-prov--hist">No evaluation results available yet.</div>')
        return
    any_entry = next(iter((ours_metrics() or historical_metrics()).values()))["metrics"]
    ref = {"src": "ref", "name": "Always predict 'unrelated'", "sub": "majority-class reference baseline",
           "acc": any_entry["majority_baseline_accuracy"], "f1": math.nan, "p": math.nan, "r": math.nan,
           "dis": 0.0, "fnc": any_entry["fnc"]["null_relative"], "raw": any_entry["fnc"]["null"]}
    T.md(f'<div class="sd-fine" style="margin:0 0 1rem">All rows are scored on the same official FNC-1 competition '
         f"test set ({any_entry['n']:,} headline–body pairs).</div>")

    best = {c: max(r[c] for r in rows) for c in ("acc", "f1", "p", "r", "dis", "fnc")}

    def cell(r, c, fmt):
        v = r[c]
        if isinstance(v, float) and math.isnan(v):
            return "<td>n/a</td>"
        cls = ' class="sd-best"' if r["src"] != "ref" and v == best.get(c) else ""
        return f"<td{cls}>{fmt(v)}</td>"

    groups = [("ours", "Trained in this project"), ("hist", "Historical · original repository (not reproduced)"),
              ("ref", "Reference")]
    body = []
    for g, title in groups:
        grp = [ref] if g == "ref" else [r for r in rows if r["src"] == g]
        if not grp:
            continue
        body.append(f'<tr class="sd-group"><td colspan="8">{T.badge(g)}&nbsp;&nbsp;{T.esc(title)}</td></tr>')
        for r in grp:
            f3 = lambda v: f"{v:.3f}"  # noqa: E731
            body.append(
                f"<tr><td>{T.esc(r['name'])}<span class=\"sd-note\">{T.esc(r['sub'])}</span></td>"
                + cell(r, "acc", f3) + cell(r, "f1", f3) + cell(r, "p", f3) + cell(r, "r", f3) + cell(r, "dis", f3)
                + cell(r, "fnc", lambda v: f"{v:.2%}") + f"<td>{r['raw']:,.2f}</td></tr>")
    T.md('<div class="sd-table-wrap"><table class="sd-table sd-table--cmp"><thead><tr><th>Model</th><th>Accuracy</th>'
         "<th>Macro-F1</th><th>Macro P</th><th>Macro R</th><th>Disagree F1</th><th>FNC-1 (rel.)</th>"
         f"<th>FNC-1 (raw)</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>"
         '<div class="sd-fine">Bold = best value in the column. FNC-1 (relative) = score / maximum possible score.</div>')

    T.md(T.h3("Macro-F1 and FNC-1 score", "dashed line = FNC-1 of always predicting 'unrelated'"))
    null = ref["fnc"]
    bars = ['<div class="sd-legend"><span><i style="background:var(--paper-2)"></i>Macro-F1</span>'
            '<span><i style="background:var(--red)"></i>FNC-1 (relative)</span></div><div class="sd-bars">']
    for r in rows:
        tag = T.badge(r["src"])
        bars.append(
            f'<div class="sd-brow"><div class="sd-bname">{T.esc(r["name"])}<small>{tag}</small></div>'
            '<div class="sd-btracks">'
            f'<div class="sd-bt sd-bt--f1"><div><i style="width:{r["f1"] * 100:.1f}%"></i></div><span>{r["f1"]:.3f}</span></div>'
            f'<div class="sd-bt sd-bt--fnc"><div><i style="width:{r["fnc"] * 100:.1f}%"></i>'
            f'<span class="sd-ref" style="left:{null * 100:.2f}%"></span></div><span>{r["fnc"]:.1%}</span></div>'
            "</div></div>")
    bars.append("</div>")
    T.md("".join(bars))

    T.md(T.h3("How to read this comparison"))
    T.md('<div class="sd-prose"><ul>'
         f"<li><strong>Accuracy is misleading on FNC-1.</strong> {ref['acc']:.0%} of test pairs are "
         f"<em>unrelated</em>, so a model that always says \"unrelated\" scores {ref['acc']:.0%} accuracy. "
         "Macro-F1 and the FNC-1 score are fairer.</li>"
         "<li><strong>Historical vs ours are not like-for-like.</strong> The original BERT was trained on an "
         "unrecorded split with an unpublished checkpoint. Ours uses a split grouped by article body, chosen on "
         "validation only.</li>"
         "<li><strong>Historical baselines are handicapped by bugs:</strong> a one-vs-rest <code>argmax</code> bug "
         "pushed uncertain rows to <em>unrelated</em>. The Count and TF-IDF Logistic Regression files were "
         "<strong>swapped</strong> in the original repository and are relabelled here. The two Bernoulli NB files "
         "are byte-identical, so they appear once.</li>"
         "<li><strong>Our original-style baselines</strong> keep the original design (bag of words over "
         "<code>headline [SEP] body</code>) with the bugs fixed. Such features cannot see <em>which</em> words come "
         "from the headline and which from the body, so they remain weak.</li>"
         "<li><strong>Our relational Logistic Regression</strong> adds headline–body similarity and overlap features "
         "(as in the official FNC-1 baseline). It is a different, stronger design and is labelled as such.</li>"
         "<li><em>Disagree</em> is the rarest class (under 3% of test pairs) and the hardest for every model.</li>"
         "</ul></div>")


# ---------------------------------------------------------------------------
# 04 Methodology
# ---------------------------------------------------------------------------
_ARROW = ('<div class="sd-arrow" aria-hidden="true"><svg viewBox="0 0 40 14"><path d="M0 7h34" stroke="currentColor" '
          'stroke-width="1.5"/><path d="M30 2l7 5-7 5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></div>')


def _methodology() -> None:
    bert = model_info("bert")
    hp = bert.get("hyperparameters", config.BERT)
    T.md(T.section_head(
        _num("methodology"), "Methodology", "From two texts to one stance",
        "Each request is a pair of texts. Both engines read the pair together and return a score for each of the "
        "four stances; the highest score becomes the verdict."))
    T.md(
        '<div class="sd-flow">'
        '<div class="sd-node"><div class="sd-n">01 · input</div><div class="sd-h4" role="heading" aria-level="4">Headline + article</div>'
        "<p>The claim and the full article text, as filed on the PREDICT monitor.</p></div>" + _ARROW +
        '<div class="sd-node"><div class="sd-n">02 · preprocessing</div><div class="sd-h4" role="heading" aria-level="4">Text preprocessing</div>'
        "<p>Validation and whitespace/Unicode normalisation; then tokenisation (BERT) or TF-IDF and "
        "relation features (LR).</p></div>" + _ARROW +
        '<div class="sd-node sd-node--engine"><div class="sd-n">03 · engine</div><div class="sd-h4" role="heading" aria-level="4">BERT / Logistic Regression</div>'
        '<div class="sd-split"><div><b>BERT</b>Fine-tuned bert-base-uncased. Default; GPU when available, '
        "else CPU.</div><div><b>Logistic Regression</b>TF-IDF + headline–body relation features. CPU fallback."
        "</div></div></div>" + _ARROW +
        '<div class="sd-node sd-node--out"><div class="sd-n">04 · output</div><div class="sd-h4" role="heading" aria-level="4">Stance classification</div>'
        "<p>Agree, disagree, discuss or unrelated, with a score for each class.</p></div>"
        "</div>")
    T.md('<div class="sd-two"><div>' + T.h3("The two prediction engines") +
         '<div class="sd-prose">'
         "<p><strong>BERT (default).</strong> <code>BertForSequenceClassification</code> on <code>bert-base-uncased</code> "
         "(12 Transformer layers, hidden size 768, about 110M parameters), official weights in safetensors format "
         f"at a pinned revision. Input <code>[CLS] headline [SEP] article [SEP]</code>, truncated to "
         f"{bert.get('max_length', config.BERT['max_length'])} tokens; segment ids tell BERT which part is the "
         "headline. The <code>[CLS]</code> representation feeds a dropout layer and a linear layer with 4 outputs; "
         "softmax turns these into the scores on the PREDICT screen.</p>"
         "<p><strong>Logistic Regression (fallback).</strong> Headline TF-IDF, body TF-IDF, cosine similarity, "
         "word/bigram overlap and refuting/hedging-word indicators. It needs no GPU, loads in seconds and is "
         "integrity-checked (SHA-256) before loading. Choose it from the engine switch on the monitor.</p>"
         "<p><strong>GPU / CPU fallback.</strong> BERT runs on the GPU when CUDA is available, otherwise on the CPU "
         "(<code>FNC_DEVICE=cpu</code> forces CPU). If the GPU runs out of memory mid-request, that prediction is "
         "recomputed on the CPU and the verdict says so.</p>"
         "</div></div><div>" + T.h3("Fine-tuning recipe", "from models/bert-fnc1/model_info.json") +
         '<dl class="sd-dl">'
         f"<dt>Optimizer</dt><dd>AdamW, learning rate {hp.get('learning_rate')}</dd>"
         f"<dt>Warm-up</dt><dd>{hp.get('warmup_steps')} steps</dd>"
         f"<dt>Weight decay</dt><dd>{hp.get('weight_decay')}</dd>"
         f"<dt>Batch</dt><dd>{hp.get('micro_batch_size')} × {hp.get('grad_accum_steps')} accumulation = "
         f"{int(hp.get('micro_batch_size', 0)) * int(hp.get('grad_accum_steps', 0))} effective</dd>"
         f"<dt>Epochs</dt><dd>{bert.get('epochs_completed', hp.get('epochs'))} of {hp.get('epochs')} "
         f"({bert.get('optimizer_steps', '?')} optimizer steps)</dd>"
         f"<dt>Precision</dt><dd>{'fp16 mixed precision' if hp.get('fp16') else 'fp32'}</dd>"
         f"<dt>Selection</dt><dd>{T.esc(bert.get('selection', 'best validation macro-F1'))}</dd>"
         f"<dt>Hardware</dt><dd>RTX 3050 laptop GPU</dd>"
         "</dl></div></div>")
    T.md(T.h3("Evaluation protocol") + '<ol class="sd-steps-list">'
         "<li><span>Download the four FNC-1 CSV files and check them against the gold labels committed in the original "
         "repository.</span></li>"
         "<li><span>Split the training data 90/10, <strong>grouped by article body</strong> and stratified by label "
         f"(seed {config.SEED}), so no article appears in both training and validation.</span></li>"
         "<li><span>Train the models. The BERT checkpoint is selected on validation macro-F1; the linear models are fit "
         "once with fixed settings, and validation is only reported for them.</span></li>"
         "<li><span>Score once on the official competition test set: accuracy, macro and weighted precision/recall/F1, "
         "per-class metrics, confusion matrix and the official FNC-1 score.</span></li>"
         "<li><span>Recompute the original repository's committed results with the same code, so the comparison is "
         "consistent.</span></li></ol>")


# ---------------------------------------------------------------------------
# 05 Dataset
# ---------------------------------------------------------------------------
def _strip(counts: dict) -> str:
    total = max(1, sum(counts.values()))
    segs, legend = [], []
    for l in config.DISPLAY_ORDER:
        share = counts[l] / total
        label = f"<span>{l}</span>" if share > 0.1 else ""
        segs.append(f'<div style="--c:{T.CLASS_COLORS[l]};width:{share * 100:.2f}%" title="{l}: {share:.1%}">{label}</div>')
        legend.append(f'<div style="--c:{T.CLASS_COLORS[l]}"><b>{l}</b>{counts[l]:,} · {share:.1%}</div>')
    return f'<div class="sd-strip">{"".join(segs)}</div><div class="sd-strip-legend">{"".join(legend)}</div>'


def _dataset() -> None:
    T.md(T.section_head(
        _num("dataset"), "Dataset", "FNC-1: headlines meet articles",
        "The Fake News Challenge stage-1 corpus pairs headlines with article bodies derived from the Emergent "
        "dataset. Each pair carries one of four stance labels."))
    T.md(f'<div class="sd-prov"><span class="sd-badge sd-badge--ref">Source</span><span>'
         f'<a href="{config.FNC1_SOURCE}" target="_blank">{config.FNC1_SOURCE}</a> · commit '
         f'<span class="sd-code">{config.FNC1_COMMIT[:12]}</span>. CSV files only; no code from that repository '
         "is run. Expected SHA-256 hashes are in <span class=\"sd-code\">data/raw/fnc1/SHA256SUMS.txt</span>.</span></div>")
    s = dataset_stats()
    if not s:
        T.md('<div class="sd-prov sd-prov--hist">The FNC-1 CSV files were not found in '
             "<code>data/raw/fnc1/</code>, so dataset statistics cannot be computed. See README 'Dataset'.</div>")
        return
    tr, te = s["train"], s["test"]
    sp = s.get("split", {})
    T.md(_ledger([
        (f"{tr['pairs']:,}", "Training pairs", f"{tr['bodies']:,} article bodies · {tr['headlines']:,} headlines"),
        (f"{te['pairs']:,}", "Test pairs", f"{te['bodies']:,} article bodies · {te['headlines']:,} headlines"),
        (f"{sp.get('train_idx', 0):,}", "Train fold", "90% · grouped by body"),
        (f"{sp.get('val_idx', 0):,}", "Validation fold", f"10% · seed {sp.get('seed', config.SEED)}"),
    ]))
    T.md(T.h3("Class imbalance · training set", "from train_stances.csv") + _strip(tr["counts"]))
    T.md(T.h3("Class imbalance · competition test set", "from competition_test_stances.csv") + _strip(te["counts"]))
    un = te["counts"]["unrelated"] / te["pairs"]
    dis = te["counts"]["disagree"] / te["pairs"]
    T.md('<div class="sd-two" style="margin-top:1.6rem"><div class="sd-prose">'
         f"<p><strong>Heavily imbalanced.</strong> {un:.0%} of test pairs are <em>unrelated</em> and only "
         f"{dis:.1%} are <em>disagree</em>. That is why the comparison leads with macro-F1 and the FNC-1 score, "
         "which reward getting the related stances right.</p>"
         "<p><strong>Grouped split.</strong> The same article body appears under many headlines, so the "
         "validation split is grouped by body: no article is seen in both training and validation.</p>"
         "</div><div class=\"sd-prose\">"
         f"<p><strong>Article length.</strong> The median article has {tr['median_words']:,} words (training) and "
         f"{te['median_words']:,} words (test). {te['share_over_400_words']:.0%} of test articles exceed 400 "
         f"words, roughly where BERT's {config.BERT['max_length']}-token window starts cutting text.</p>"
         "<p><strong>Domain and time.</strong> English news collected for the 2017 challenge. The test set covers "
         "topics not seen in training.</p></div></div>")


# ---------------------------------------------------------------------------
# 06 Limitations & project information
# ---------------------------------------------------------------------------
def _limitations() -> None:
    T.md(T.section_head(
        _num("limitations"), "Limitations & project information", "What the desk cannot tell you",
        "A stance verdict is a useful signal for triage, not a judgement on truth. These are the known limits of "
        "the models and of this measurement."))
    ours = ours_metrics()
    gap = ""
    if "bert_fnc1" in ours and ours["bert_fnc1"]["meta"].get("validation_metrics"):
        vf1 = ours["bert_fnc1"]["meta"]["validation_metrics"]["macro"]["f1"]
        tf1 = ours["bert_fnc1"]["metrics"]["macro"]["f1"]
        gap = (f"<li><strong>Validation vs test gap.</strong> Validation macro-F1 {vf1:.3f} vs test {tf1:.3f}. "
               "The test set covers unseen topics.</li>")
    dis = ""
    if ours:
        parts = [f"{ours[k]['meta'].get('display_name', k)}: {ours[k]['metrics']['per_class']['disagree']['recall']:.3f}"
                 for k in ("bert_fnc1", "lr_relational") if k in ours]
        dis = f" Test recall on <em>disagree</em>: {T.esc('; '.join(parts))}." if parts else ""
    hist = historical_metrics()
    hist_acc = (f" ({fmt_pct(hist['BERT']['metrics']['accuracy'])} accuracy, recomputed from its committed predictions)"
                if "BERT" in hist else "")
    T.md('<div class="sd-prose"><ul>'
         "<li><strong>Not a fact-checker.</strong> An article that <em>agrees</em> with a headline can still be "
         "false, and one that <em>disagrees</em> can be wrong. The desk never labels claims as true or false.</li>"
         f"<li><strong>Minority classes are hard.</strong> <em>Disagree</em> is under 3% of test pairs and its recall "
         f"is the weakest point of every model.{dis}</li>"
         "<li><strong>Confidence is not calibrated.</strong> The BERT model often reports 99%+ confidence, even on "
         "some wrong predictions. Scores are softmax outputs, not calibrated probabilities.</li>"
         f"<li><strong>Long articles are truncated</strong> to {config.BERT['max_length']} tokens for BERT "
         "(headline + start of the article), so stance expressed late in an article can be missed. The verdict "
         "warns when this happens.</li>"
         "<li><strong>Domain and time.</strong> FNC-1 contains English news articles collected for the 2017 "
         "challenge. Performance on newer topics, other languages or other writing styles is unknown.</li>"
         f"{gap}"
         "<li><strong>Historical results cannot be reproduced.</strong> The original checkpoint is private. Our "
         "numbers come from our own training run and are a different measurement, not a reproduction of the "
         f"original BERT result{hist_acc}.</li>"
         "<li><strong>LR engine persistence</strong> uses joblib (pickle). It is written by our own script and "
         "SHA-256-verified before loading; never load such files from other sources.</li>"
         "<li><strong>No checkpoint resume</strong> in <code>train_bert.py</code>; the time budget stops training "
         "gracefully and saves the best model instead.</li>"
         "<li>The original repository's \"prompt-based\" framing is not implemented; it was standard "
         "fine-tuning.</li>"
         "</ul></div>")
    T.md(T.h3("Project background & credits") +
         '<div class="sd-two"><div class="sd-prose">'
         "<p>This application is a working, audited re-implementation of "
         '<a href="https://github.com/zeyongj/Fake-News-Stance-Detection" target="_blank">'
         "zeyongj/Fake-News-Stance-Detection</a> for the FNC-1 headline–body stance task. The original "
         "repository is used read-only: its committed prediction files are re-scored here, and nothing in it is "
         "modified.</p>"
         "<p>All models shown as <em>trained here</em> were trained and evaluated in this project. Every deviation "
         "from the original is recorded in <code>docs/DECISIONS.md</code>, and the fixes in "
         "<code>docs/CHANGELOG.md</code>.</p></div>"
         '<div><dl class="sd-dl">'
         f'<dt>Dataset</dt><dd><a href="{config.FNC1_SOURCE}" target="_blank">Fake News Challenge FNC-1</a></dd>'
         f"<dt>Base model</dt><dd>{T.esc(config.PRETRAINED_BERT_SOURCE)} @ {config.PRETRAINED_BERT_REVISION[:12]}</dd>"
         "<dt>Original code</dt><dd>zeyongj/Fake-News-Stance-Detection (read-only reference)</dd>"
         "<dt>Libraries</dt><dd>PyTorch, Hugging Face Transformers, scikit-learn, Streamlit</dd>"
         "<dt>Runs</dt><dd>Locally and offline; models load from local files only</dd>"
         "</dl></div></div>")
