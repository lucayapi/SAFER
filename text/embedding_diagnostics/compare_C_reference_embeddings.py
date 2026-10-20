#!/usr/bin/env python3
"""Compare C-only cosine neighbours for one identical reference in two Qwen spaces."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "text/dataset/families/btp/data_btp_building_construction_and_masonry.csv"
NORMAL = ROOT / "text/embeddings/families/btp/Qwen3-Embedding-0.6B_btp_building_construction_and_masonry.csv"
INSTRUCTED = ROOT / "text/embeddings/btp_building_construction_and_masonry/Qwen3-Embedding-0.6B_btp_C_health_consequence_instruct.csv"
REFERENCE_ID = 37882
OUT = ROOT / "text/embedding_diagnostics/btp_building_construction_masonry_C_reference_comparison"

def normalized(df, role_units):
    frame = role_units.merge(df, left_on="fact_id", right_on="doc_id", how="left", validate="one_to_one")
    if frame.doc_id.isna().any():
        raise ValueError(f"Missing embedding for {int(frame.doc_id.isna().sum())} C units")
    cols = [c for c in df.columns if c.startswith("dim_")]
    x = frame[cols].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite embedding values")
    norms = np.linalg.norm(x, axis=1)
    if (norms == 0).any():
        raise ValueError("Zero embedding vector")
    x /= norms[:, None]
    return frame, x

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(UNITS, dtype={"fact_id":"Int64", "accident_id":"string"})
    c = units.loc[units.pred_label.eq("C"), ["fact_id","accident_id","sentence"]].copy()
    c["fact_id"] = c.fact_id.astype("Int64")
    anchor_rows = c.loc[c.fact_id.eq(REFERENCE_ID)]
    if len(anchor_rows) != 1:
        raise ValueError(f"Expected one C reference {REFERENCE_ID}, got {len(anchor_rows)}")
    anchor = anchor_rows.iloc[0]
    outputs=[]; audits=[]
    for name, path in [("normal",NORMAL),("instructed",INSTRUCTED)]:
        emb = pd.read_csv(path, dtype={"doc_id":"Int64"})
        if emb.doc_id.duplicated().any(): raise ValueError(f"Duplicate doc_id in {path}")
        frame,x=normalized(emb,c)
        ref_idx=int(np.flatnonzero(frame.fact_id.to_numpy()==REFERENCE_ID)[0])
        scores=x @ x[ref_idx]
        eligible=frame.accident_id.astype(str).to_numpy()!=str(anchor.accident_id)
        eligible[ref_idx]=False
        scores[~eligible]=-np.inf
        idx=np.argpartition(-scores,9)[:10]
        idx=idx[np.argsort(-scores[idx])]
        for rank,j in enumerate(idx,1):
            z=frame.iloc[int(j)]
            outputs.append({"embedding_type":name,"reference_fact_id":REFERENCE_ID,
                "reference_accident_id":str(anchor.accident_id),"reference_text":anchor.sentence,
                "rank":rank,"neighbour_fact_id":int(z.fact_id),"neighbour_accident_id":str(z.accident_id),
                "cosine_similarity":float(scores[j]),"neighbour_text":z.sentence})
        audits.append({"embedding_type":name,"file":str(path.relative_to(ROOT)),"rows":len(emb),
            "dimensions":len([col for col in emb if col.startswith('dim_')]),"C_units":len(c),
            "C_coverage":len(frame),"anchor_id":REFERENCE_ID,"anchor_text":anchor.sentence,
            "neighbors":10,"same_accident_excluded":True})
    out=pd.DataFrame(outputs)
    out.to_csv(OUT/"C_reference_top10_comparison.csv",index=False,encoding="utf-8-sig")
    (OUT/"audit.json").write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding="utf-8")
    print(out[["embedding_type","rank","cosine_similarity","neighbour_text"]].to_string(index=False))
    print("AUDIT",json.dumps(audits,ensure_ascii=False))

if __name__=="__main__": main()
